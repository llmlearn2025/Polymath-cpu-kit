/* qwen36_kn_cuda.h — knowledge-expert packs (coli_kn.h) on qwen36's CUDA chain (moe-knowledge).
 * Included by qwen36.c in a COLI_CUDA build, after qwen36_cuda_chain.h.
 *
 * On a swap the pack goes to the card (its own buffers: the experts in the int8-rows layout the
 * chain's cc_moe reads, fmt 1, their table, the gate biases, each layer's gate matrix as an f32
 * tensor). Every path a layer's routed experts take on this chain (the device's experts, the
 * hybrid host share, the host's step for a prompt chunk) ends in q36cc_combine with the routed
 * sum in a device buffer and the MoE input still in ch->h2; there, for a layer with knowledge
 * experts, the chain runs, on its own stream and with its own kernels:
 *     logits = gate_w . h2;  g = sigmoid(logits + gate_b)   (cc_matmul, cc_route kind 1, all K)
 *     routed' = routed + sum_j g_j * expert_j(h2)            (cc_moe, the routed sum its base row)
 * and combines routed' instead, which is coli_kn_apply's arithmetic on the CPU. Layers without
 * knowledge experts, and the model without a pack, run exactly as before.
 *
 * The new pack's device copy is built BEFORE the swap (coli_kn_set -> prepare); if anything fails
 * the swap is refused and the old pack stays, on the card and on the CPU. */
#ifndef QWEN36_KN_CUDA_H
#define QWEN36_KN_CUDA_H

#define Q36KN_PAD(b) (((size_t)(b) + 255) / 256 * 256)

typedef struct {
    int L, Ktot, Kmax;
    int *K, *off;
    CcBuf *ex, *tab, *bias;
    ColiCudaTensor **gt;
    size_t bytes;
} Q36KnDev;
static Q36KnDev g_q36kn, g_q36kn_next;
static CcBuf *g_q36kn_lg, *g_q36kn_ids, *g_q36kn_wt, *g_q36kn_hid, *g_q36kn_out;   /* scratch, kept */
static int g_q36kn_rows;

static Q36CChain *q36kn_chain(void *mv) {
    Q36CChain *ch = mv ? q36cc_of((Model *)mv) : NULL;
    return ch && ch->ok && !ch->failed ? ch : NULL;
}
static void q36kn_free(Q36KnDev *d) {
    if (d->gt) for (int l = 0; l < d->L; l++) if (d->gt[l]) coli_cuda_tensor_free(d->gt[l]);
    cc_free(d->ex); cc_free(d->tab); cc_free(d->bias);
    free(d->gt); free(d->K); free(d->off);
    memset(d, 0, sizeof *d);
}
static size_t q36kn_stride(int D, int I) {
    const size_t ng = (size_t)I * D;
    return 2 * Q36KN_PAD(ng) + Q36KN_PAD((size_t)D * I) + Q36KN_PAD((size_t)(2 * I + D) * sizeof(float));
}

/* the chain's scratch for `rows` rows (called from q36cc_bufs, before any frame) */
static int q36_kn_cuda_bufs(Q36CChain *ch, Model *m, int rows) {
    (void)ch;
    const Q36KnDev *d = g_q36kn_next.Ktot ? &g_q36kn_next : &g_q36kn;
    int K = d->Kmax > g_q36kn.Kmax ? d->Kmax : g_q36kn.Kmax;
    if (!K) return 1;
    const int D = m->c.hidden, I = m->c.inter;
    const size_t r = (size_t)rows;
    int ok = cc_reserve(&g_q36kn_lg, r * K * 4, CC_DEV) && cc_reserve(&g_q36kn_ids, r * K * 4, CC_DEV) &&
             cc_reserve(&g_q36kn_wt, r * K * 4, CC_DEV) && cc_reserve(&g_q36kn_out, r * D * 4, CC_DEV) &&
             cc_reserve(&g_q36kn_hid, cc_moe_hid_floats(rows, K, D, I, K) * 4, CC_DEV);
    if (ok && rows > g_q36kn_rows) g_q36kn_rows = rows;
    return ok;
}

/* Layer i's routed sum for n rows, with the pack's knowledge experts added where the layer has
 * them: `routed` itself, or the chain's knowledge output buffer. NULL: a failure. */
static CcBuf *q36_kn_cuda_layer(Q36CChain *ch, Model *m, int i, int n, CcBuf *routed) {
    const Q36KnDev *d = &g_q36kn;
    if (!d->Ktot || i >= d->L || !d->K[i]) return routed;
    if (n > g_q36kn_rows) return NULL;   /* q36cc_bufs reserves for every chunk it sizes */
    const int D = m->c.hidden, I = m->c.inter, K = d->K[i], o = d->off[i];
    CcRoute ro = {n, K, K, 1, 0, K, o, 2, 0, K, 0, K, 1, 1, 0, 1.f, 1.f, 0.f};
    CcMoe mp = {n, K, D, I, 1, 0, 0, 3, o, 0, D, 0, K, 0, K, 0, D, 0, K, 0, 0};
    int ok = cc_matmul(d->gt[i], ch->h2, 0, g_q36kn_lg, 0, n) &&
             cc_route(g_q36kn_lg, d->bias, g_q36kn_ids, g_q36kn_wt, &ro) &&
             cc_moe(d->tab, ch->h2, g_q36kn_ids, g_q36kn_wt, g_q36kn_hid, g_q36kn_out, routed, &mp);
    return ok ? g_q36kn_out : NULL;
}

/* coli_kn_set's prepare: nw's device copy into g_q36kn_next (nothing to do off the chain) */
static int q36_kn_cuda_prepare(void *mv, const ColiKn *nw) {
    Q36CChain *ch = q36kn_chain(mv);
    q36kn_free(&g_q36kn_next);
    if (!ch || !nw) return 1;
    Model *m = mv;
    const int L = nw->L, D = nw->D, I = nw->I;
    if (D > CC_MOE_DIM_MAX || I > CC_MOE_DIM_MAX || D % 32 || I % 32) {
        fprintf(stderr, "[OVERLAY] CUDA: experts of %d x %d are not in the chain's kernels\n", D, I); return 0; }
    Q36KnDev *d = &g_q36kn_next;
    d->L = L;
    d->K = calloc(L, sizeof(int)); d->off = calloc(L, sizeof(int)); d->gt = calloc(L, sizeof(ColiCudaTensor *));
    if (!d->K || !d->off || !d->gt) { q36kn_free(d); return 0; }
    for (int l = 0; l < L; l++) {
        d->K[l] = nw->kn[l].K; d->off[l] = d->Ktot; d->Ktot += nw->kn[l].K;
        if (nw->kn[l].K > d->Kmax) d->Kmax = nw->kn[l].K;
        if (nw->kn[l].K > CC_ROUTE_KMAX) {
            fprintf(stderr, "[OVERLAY] CUDA: layer %d has %d knowledge experts, the chain takes %d\n", l, nw->kn[l].K, CC_ROUTE_KMAX);
            q36kn_free(d); return 0; }
    }
    const size_t st = q36kn_stride(D, I), ng = (size_t)I * D, nd = (size_t)D * I;
    const size_t ou = Q36KN_PAD(ng), od = 2 * Q36KN_PAD(ng), osc = od + Q36KN_PAD(nd);
    const size_t nw8 = 3 * ng, ns = 2 * (size_t)I + D;
    const int T = d->Ktot;
    double t0 = tm_now();
    uint8_t *buf = calloc(1, st);
    float *bias = malloc(sizeof(float) * (size_t)T);
    CcExpert *tb = calloc((size_t)T, sizeof(CcExpert));
    d->ex = cc_buf((size_t)T * st, CC_DEV);
    d->tab = cc_buf((size_t)T * sizeof(CcExpert), CC_DEV);
    d->bias = cc_buf((size_t)T * sizeof(float), CC_DEV);
    const char *base = d->ex ? (const char *)cc_dev_addr(d->ex) : NULL;
    int ok = buf && bias && tb && d->ex && d->tab && d->bias && base;
    for (int l = 0; ok && l < L; l++) {
        const ColiKnLayer *k = &nw->kn[l];
        if (!k->K) continue;
        for (int j = 0; ok && j < k->K; j++) {
            const int t = d->off[l] + j;
            const int8_t *w = k->w8 + (size_t)j * nw8;
            const float *sc = k->sc + (size_t)j * ns;
            memcpy(buf, w, ng); memcpy(buf + ou, w + ng, ng); memcpy(buf + od, w + 2 * ng, nd);
            memcpy(buf + osc, sc, ns * sizeof(float));
            ok = cc_begin() && cc_write(d->ex, (size_t)t * st / 4, buf, st) && cc_submit(1);
            const char *g = base + (size_t)t * st;
            const float *gsc = (const float *)(g + osc);
            tb[t].g = g; tb[t].u = g + ou; tb[t].d = g + od;
            tb[t].gs = gsc; tb[t].us = gsc + I; tb[t].ds = gsc + 2 * I;
            bias[t] = k->gb[j];
        }
        ok = ok && coli_cuda_tensor_upload(&d->gt[l], k->gw, NULL, 0, D, k->K, ch->dev);
    }
    ok = ok && cc_begin() && cc_write(d->tab, 0, tb, (size_t)T * sizeof(CcExpert)) &&
         cc_write(d->bias, 0, bias, (size_t)T * sizeof(float)) && cc_submit(1);
    ok = ok && q36_kn_cuda_bufs(ch, m, ch->rows > 0 ? ch->rows : 1);
    d->bytes = (size_t)T * (st + sizeof(CcExpert) + sizeof(float) + (size_t)D * sizeof(float));
    free(buf); free(bias); free(tb);
    if (!ok) { fprintf(stderr, "[OVERLAY] CUDA: the pack did not reach the card%s\n", cc_lost() ? " (device lost)" : "");
               q36kn_free(d); return 0; }
    fprintf(stderr, "[OVERLAY] CUDA: 0 expert(s) re-uploaded to the card, %d knowledge expert(s) on the card "
                    "(%.1f MiB; %d before) in %.1f ms\n", T, d->bytes / 1048576.0, g_q36kn.Ktot, tm_now() - t0);
    return 1;
}

/* coli_kn_set's commit: the prepared copy becomes current (nothing on the card reads the old one:
 * every frame is waited for before a forward returns) */
static void q36_kn_cuda_commit(void *mv, const ColiKn *nw) {
    Q36CChain *ch = q36kn_chain(mv);
    if (ch) { if (!cc_lost()) cc_finish(); }
    int had = g_q36kn.Ktot;
    q36kn_free(&g_q36kn);
    g_q36kn = g_q36kn_next;
    memset(&g_q36kn_next, 0, sizeof g_q36kn_next);
    if (ch && !nw && had)
        fprintf(stderr, "[OVERLAY] CUDA: 0 expert(s) re-uploaded to the card, 0 knowledge expert(s) on the card "
                        "(0.0 MiB; %d before) in 0.0 ms\n", had);
}

__attribute__((constructor)) static void q36_kn_cuda_register(void) {
    g_kn_dev_prepare = q36_kn_cuda_prepare;
    g_kn_dev_commit = q36_kn_cuda_commit;
}
#endif
