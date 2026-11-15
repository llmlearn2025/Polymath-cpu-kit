/* olmoe_overlay_cuda.h — overlays on OLMoE's CUDA chain (moe-knowledge). Included by olmoe.c in a
 * COLI_CUDA build, after olmoe_cuda_chain.h.
 *
 * Modified experts (v1, method A). The chain holds every routed expert in one device buffer
 * (g_occ->ex), expert (layer, e) at ((layer * E) + e) * ex_stride, written once at start by
 * occ_expert_bytes(). That function reads the expert through load_expert_merged(), which resolves
 * the ACTIVE overlay first, so a swap re-uploads exactly the affected experts (old cells and new
 * cells) with the same function: overlay bytes in, base bytes back on removal. The routing table
 * points into the same buffer and does not change.
 *
 * Knowledge experts (v2, B-slot). An overlay's knowledge experts go to the card on a swap, in a
 * buffer of their own (the same int8-rows layout and table entries as the routed experts), with
 * each layer's gate matrix [K, hidden] as a resident f32 tensor and the gate biases beside them.
 * In the chain, after a layer's routed experts (olmoe_cuda_chain.h, occ_layer), a layer that has
 * knowledge experts runs, on the same stream and with the chain's own kernels:
 *     logits = gate_w . h2                      (cc_matmul, h2 = the MoE input)
 *     g      = sigmoid(logits + gate_b)         (cc_route: kind 1, the bias added to the logit,
 *                                               every one of the K picked, no normalisation)
 *     out    = routed_sum + sum_j g_j * expert_j(h2)   (cc_moe: hy = the routed sum)
 * which is olm_kn_apply's arithmetic on the CPU. The router, the routed experts and their table
 * are untouched; layers without knowledge experts run exactly as before.
 *
 * Swaps happen between forward passes (olm_overlay_poll at the head of step()); every frame the
 * chain submits is waited for, so nothing on the card reads the buffers a swap frees. If any
 * upload fails, the host's KV cache is brought up to date from the card and the chain is marked
 * failed: the CPU, whose slots and knowledge experts already hold the new overlay, takes over:
 * never a mix of old and new experts. */
#ifndef OLMOE_OVERLAY_CUDA_H
#define OLMOE_OVERLAY_CUDA_H

typedef struct {
    int L;                        /* layers with a K entry (n_layers), 0 = no knowledge experts on the card */
    int *K, *off;                 /* per layer: its knowledge experts, and its first entry in tab/bias */
    int Ktot, Kmax;
    CcBuf *ex, *tab, *bias;       /* expert bytes, CcExpert entries, gate biases */
    ColiCudaTensor **gt;          /* per layer: gate_w as a [K x hidden] f32 tensor */
    size_t bytes;
} OlmKnDev;
static OlmKnDev g_knd;
static CcBuf *g_knd_lg, *g_knd_ids, *g_knd_wt, *g_knd_hid;   /* scratch, kept across overlays */

static int olm_ov_chain_on(void) { return g_occ && g_occ->ok && !g_occ->failed; }

static void olm_knd_free(void) {
    if (g_knd.gt) for (int l = 0; l < g_knd.L; l++) if (g_knd.gt[l]) coli_cuda_tensor_free(g_knd.gt[l]);
    cc_free(g_knd.ex); cc_free(g_knd.tab); cc_free(g_knd.bias);
    free(g_knd.gt); free(g_knd.K); free(g_knd.off);
    memset(&g_knd, 0, sizeof g_knd);
}

/* the chain's scratch for `rows` rows: the knowledge experts' logits, picks, gates and hidden rows */
static int olm_kn_cuda_bufs(OccChain *ch, Model *m, int rows) {
    (void)ch;
    if (!g_knd.Ktot) return 1;
    const int D = m->c.hidden, I = m->c.inter, K = g_knd.Kmax;
    const size_t r = (size_t)rows;
    return occ_res(&g_knd_lg, r * K, CC_DEV) && occ_res(&g_knd_ids, r * K, CC_DEV) && occ_res(&g_knd_wt, r * K, CC_DEV) &&
           occ_res(&g_knd_hid, cc_moe_hid_floats(rows, K, D, I, K), CC_DEV);
}

/* Layer i's MoE output for n rows: ch->y (the routed sum) when the layer has no knowledge
 * experts; else ch->tmp = ch->y + each knowledge expert's gated output. NULL: a failure. */
static CcBuf *olm_kn_cuda_layer(OccChain *ch, Model *m, int i, int n) {
    if (!g_knd.Ktot || i >= g_knd.L || !g_knd.K[i]) return ch->y;
    const int D = m->c.hidden, I = m->c.inter, K = g_knd.K[i], o = g_knd.off[i];
    /* S, E, K, kind (1 sigmoid), l_off, l_row, b_off, has_bias (2: to the logit), i_off, i_row,
     * w_off, w_row, n_group, topk_group, norm (0: none), temp, scale, norm_eps */
    CcRoute ro = {n, K, K, 1, 0, K, o, 2, 0, K, 0, K, 1, 1, 0, 1.f, 1.f, 0.f};
    /* S, K, D, I, fmt (1 int8 rows), gs, act, parts (3), t_off, x_off, x_row, i_off, i_row,
     * w_off, w_row, y_off, y_row, h_off (0: hy's rows first), E, planar, fmt_d */
    CcMoe mp = {n, K, D, I, 1, 0, 0, 3, o, 0, D, 0, K, 0, K, 0, D, 0, K, 0, 0};
    int ok = cc_matmul(g_knd.gt[i], ch->h2, 0, g_knd_lg, 0, n) &&
             cc_route(g_knd_lg, g_knd.bias, g_knd_ids, g_knd_wt, &ro) &&
             cc_moe(g_knd.tab, ch->h2, g_knd_ids, g_knd_wt, g_knd_hid, ch->tmp, ch->y, &mp);
    return ok ? ch->tmp : NULL;
}

/* The overlay's knowledge experts onto the card (nw NULL or without them: none). 0: a failure. */
static int olm_knd_build(Model *m, const OlmOverlay *nw) {
    olm_knd_free();
    if (!nw || !nw->kn || nw->nkn < 1) return 1;
    OccChain *ch = g_occ;
    const Cfg *c = &m->c; const int L = c->n_layers, D = c->hidden, I = c->inter;
    const size_t st = ch->ex_stride, ng = (size_t)I * D, nd = (size_t)D * I;
    const size_t ou = OCC_PAD(ng), od = 2 * OCC_PAD(ng), osc = od + OCC_PAD(nd);
    if (osc + OCC_PAD((size_t)(2 * I + D) * sizeof(float)) > st) return 0;
    g_knd.L = L;
    g_knd.K = calloc(L, sizeof(int)); g_knd.off = calloc(L, sizeof(int)); g_knd.gt = calloc(L, sizeof(ColiCudaTensor *));
    if (!g_knd.K || !g_knd.off || !g_knd.gt) return 0;
    for (int l = 0; l < L && l < nw->nkn_layers; l++) {
        g_knd.K[l] = nw->kn[l].K; g_knd.off[l] = g_knd.Ktot; g_knd.Ktot += nw->kn[l].K;
        if (nw->kn[l].K > g_knd.Kmax) g_knd.Kmax = nw->kn[l].K;
    }
    const int T = g_knd.Ktot;
    uint8_t *buf = calloc(1, st);
    float *bias = malloc(sizeof(float) * (size_t)T);
    CcExpert *tb = calloc((size_t)T, sizeof(CcExpert));
    g_knd.ex = cc_buf((size_t)T * st, CC_DEV);
    g_knd.tab = cc_buf((size_t)T * sizeof(CcExpert), CC_DEV);
    g_knd.bias = cc_buf((size_t)T * sizeof(float), CC_DEV);
    const char *base = g_knd.ex ? (const char *)cc_dev_addr(g_knd.ex) : NULL;
    int ok = buf && bias && tb && g_knd.ex && g_knd.tab && g_knd.bias && base;
    for (int l = 0; ok && l < L; l++) {
        const OlmKn *k = &nw->kn[l];
        if (!k->K) continue;
        for (int j = 0; ok && j < k->K; j++) {
            const int t = g_knd.off[l] + j;
            const int8_t *w = k->w8 + (size_t)j * 3 * ng;               /* [gate | up | down] */
            const float *sc = k->sc + (size_t)j * (2 * (size_t)I + D);  /* [I | I | D] */
            memcpy(buf, w, ng); memcpy(buf + ou, w + ng, ng); memcpy(buf + od, w + 2 * ng, nd);
            memcpy(buf + osc, sc, (size_t)(2 * I + D) * sizeof(float));
            ok = cc_begin() && cc_write(g_knd.ex, (size_t)t * st / 4, buf, st) && cc_submit(1);
            const char *g = base + (size_t)t * st;
            const float *gsc = (const float *)(g + osc);
            tb[t].g = g; tb[t].u = g + ou; tb[t].d = g + od;
            tb[t].gs = gsc; tb[t].us = gsc + I; tb[t].ds = gsc + 2 * I;
            bias[t] = k->gb[j];
        }
        ok = ok && coli_cuda_tensor_upload(&g_knd.gt[l], k->gw, NULL, 0, D, k->K, ch->dev);
    }
    ok = ok && cc_begin() && cc_write(g_knd.tab, 0, tb, (size_t)T * sizeof(CcExpert)) &&
         cc_write(g_knd.bias, 0, bias, (size_t)T * sizeof(float)) && cc_submit(1);
    ok = ok && olm_kn_cuda_bufs(ch, m, ch->rows > 0 ? ch->rows : 1);
    g_knd.bytes = (size_t)T * (st + sizeof(CcExpert) + sizeof(float) + (size_t)D * sizeof(float));
    free(buf); free(bias); free(tb);
    if (!ok) olm_knd_free();
    return ok;
}

/* the chain off for good, the host's KV cache first made the card's (or rebuilt on the CPU) */
static void olm_ov_cuda_fail(Model *m, const char *why) {
    OccChain *ch = g_occ;
    int t = ch->cap;
    for (int i = 0; i < m->c.n_layers; i++) if (ch->kv_valid && ch->kv_valid[i] < t) t = ch->kv_valid[i];
    if (t > 0 && (cc_lost() || !occ_sync(ch, m, t))) occ_recover(m, t);
    ch->failed = 1;
    olm_knd_free();
    fprintf(stderr, "[OVERLAY] CUDA: %s failed; the CPU runs from now on (its experts are current)\n", why);
}

static int olm_ov_cuda_check(void *mv, const void *nwv) {
    const Model *m = mv;
    const OlmOverlay *nw = nwv;
    if (!olm_ov_chain_on() || !nw || nw->nkn < 1) return 1;
    for (int l = 0; l < nw->nkn_layers; l++)
        if (nw->kn[l].K > CC_ROUTE_KMAX) {
            fprintf(stderr, "[OVERLAY] refused: layer %d has %d knowledge experts; the CUDA chain takes up to %d a layer. "
                            "Run it with COLI_CUDA_CHAIN=0 (CPU)\n", l, nw->kn[l].K, CC_ROUTE_KMAX);
            return 0;
        }
    if (m->c.hidden > CC_MOE_DIM_MAX || m->c.inter > CC_MOE_DIM_MAX) {
        fprintf(stderr, "[OVERLAY] refused: knowledge experts of this size are not in the CUDA chain's kernels\n");
        return 0;
    }
    return 1;
}

static void olm_ov_cuda_sync(void *mv, const void *oldv, const void *nwv) {
    if (!olm_ov_chain_on()) return;
    Model *m = mv;
    const OlmOverlay *old = oldv, *nw = nwv;
    OccChain *ch = g_occ;
    const int E = m->c.n_experts;
    uint8_t *buf = malloc(ch->ex_stride), *tmp = malloc(ch->ex_stride);
    int n = 0, ok = buf && tmp;
    double t0 = now_s();
    for (int pass = 0; ok && pass < 2; pass++) {     /* modified experts: old cells back to base, new cells in */
        const OlmOverlay *src = pass ? nw : old;
        for (int i = 0; ok && src && i < src->ncells; i++) {
            int l = src->cells[2 * i], e = src->cells[2 * i + 1];
            if (pass && olm_ov_has(old, l, e)) continue;
            ok = occ_expert_bytes(m, l, e, buf, ch->ex_stride, tmp) &&
                 cc_begin() && cc_write(ch->ex, ((size_t)l * E + e) * ch->ex_stride / 4, buf, ch->ex_stride) &&
                 cc_submit(1);
            n += ok;
        }
    }
    free(buf); free(tmp);
    if (!ok) { olm_ov_cuda_fail(m, "re-upload of the swapped experts"); return; }
    const int had = g_knd.Ktot;
    if (!olm_knd_build(m, nw)) { olm_ov_cuda_fail(m, "upload of the knowledge experts"); return; }
    fprintf(stderr, "[OVERLAY] CUDA: %d expert(s) re-uploaded to the card, %d knowledge expert(s) on the card "
                    "(%.1f MiB; %d before) in %.1f ms\n",
            n, g_knd.Ktot, g_knd.bytes / 1048576.0, had, (now_s() - t0) * 1e3);
}

__attribute__((constructor)) static void olm_ov_cuda_register(void) {
    g_ov_dev_check = olm_ov_cuda_check;
    g_ov_dev_sync = olm_ov_cuda_sync;
}
#endif
