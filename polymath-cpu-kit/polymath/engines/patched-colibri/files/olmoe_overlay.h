/* olmoe_overlay.h — versioned expert overlays for the OLMoE engine (colibri-olmoe-overlay-v1).
 *
 * An overlay is a directory written by moe-knowledge/export/to_colibri_overlay.py:
 *   overlay.safetensors  model.layers.L.mlp.experts.E.merged_weight (I8) + .qs (F32),
 *                        byte-for-byte the layout tools/convert_olmoe_merged.py gives the base
 *   overlay.json         {"format":"colibri-olmoe-overlay-v1","version":N,"base_fingerprint":"..",
 *                         "cells":[[L,E],...], ...}
 *
 * Resolution: an expert named in the ACTIVE overlay is read from the overlay, every other expert
 * from the base model. Base files are never written. Deactivating the overlay restores the base
 * bytes exactly (the affected resident slots are re-read from the base).
 *
 * Activation:
 *   OLM_OVERLAY=<dir>          at start
 *   kill -HUP <pid>            re-reads the file named by OLM_OVERLAY_CTL (a directory path, or
 *                              empty to deactivate) at the next forward pass: a live swap
 *
 * Swap protocol (olm_overlay_set), all under g_pilot_mx:
 *   1. wait until no slot anywhere is being loaded (eid < -1) and no affected slot is busy
 *      (expert_put broadcasts while a swap waits), so no reader holds the old overlay's bytes
 *      or its file descriptors;
 *   2. publish the new overlay pointer and bump the epoch;
 *   3. re-read every RESIDENT affected slot (old cells and new cells) from its new source, in
 *      place, so no stale expert can be served from the RAM cache or the pinned hot store;
 *   then, outside the lock, close the old overlay's files.
 * Loaders read g_ov only after cache_reserve, i.e. after step 1 has proven no older loader is in
 * flight, so a loader can never see a freed overlay.
 *
 * Format v2 (colibri-olmoe-overlay-v2) adds KNOWLEDGE EXPERTS (method B-slot): extra experts
 * per layer, held in RAM, that run on every token next to the routed experts and add
 *     out += sigmoid(gate_w[j] . x + gate_b[j]) * expert_j(x)
 * Tensors: model.layers.L.mlp.knowledge.J.merged_weight (I8) / .qs (F32), and per layer
 * model.layers.L.mlp.knowledge.gate_w (F32 [K,hidden]) and .gate_b (F32 [K]). Manifest key
 * "knowledge": [[L, K], ...]. The router is untouched, so routing, the expert cache and the
 * streamed experts are exactly as without the overlay. Knowledge experts swap with the overlay.
 *
 * Scope: the CPU expert path, with or without the GPU running the dense part (see olm_overlay_set), and
 * colibri's CUDA chain (olmoe_overlay_cuda.h re-uploads swapped experts and runs knowledge experts on
 * the card). A Vulkan expert tier or layer chain keeps its own expert copies: refused there.
 * Added experts (a larger n_experts) are out of scope: the engine sizes its tables at load.
 */
#ifndef OLMOE_OVERLAY_H
#define OLMOE_OVERLAY_H
#include <signal.h>
#include "coli_packs_status.h"   /* the status file a client (the Hermes plugin) checks */

#define OLM_OV_FORMAT "colibri-olmoe-overlay-v1"

#define OLM_OV_FORMAT2 "colibri-olmoe-overlay-v2"

typedef struct {         /* one layer's knowledge experts (B-slot), resident in RAM */
    int K;
    int8_t *w8;          /* K x merged int8 [gate I*D | up I*D | down D*I] */
    float *sc;           /* K x row scales [I | I | D] */
    float *gw, *gb;      /* gates: [K, D] and [K] */
    Slot *slots;         /* K views into w8/sc in the engine's Slot form (moe_expert_row) */
} OlmKn;

typedef struct {
    shards S;
    char dir[1024];
    long long version;
    int ncells;
    int *cells;          /* ncells pairs (layer, expert) */
    OlmKn *kn;           /* [n_layers], NULL when the overlay has no knowledge experts */
    int nkn_layers;      /* length of kn */
    int nkn;             /* total knowledge experts */
} OlmOverlay;

static OlmOverlay *g_ov = NULL;          /* active overlay; NULL = base only */
/* Device hooks, set by olmoe_overlay_cuda.h in a CUDA build: check whether the running device
 * path can take an overlay, and re-upload the swapped experts to the card after a swap. */
static int  (*g_ov_dev_check)(void *m, const void *nw) = NULL;
static void (*g_ov_dev_sync)(void *m, const void *old, const void *nw) = NULL;
static unsigned long long g_ov_epoch = 0;
static int g_ov_waiting = 0;             /* a swap is draining readers (expert_put broadcasts) */
static volatile sig_atomic_t g_ov_reload = 0;

/* The same architecture fields, in the same order, as export/to_colibri_overlay.py. */
static unsigned long long olm_ov_fingerprint(const Cfg *c) {
    unsigned long long h = 1469598103934665603ULL;
    long long v[6] = {c->hidden, c->inter, c->n_layers, c->n_experts, c->topk, c->vocab};
    for (int k = 0; k < 6; k++)
        for (int i = 0; i < 8; i++) { h ^= (unsigned long long)((v[k] >> (8 * i)) & 0xff); h *= 1099511628211ULL; }
    return h;
}

/* Source for one expert tensor: the active overlay if it holds the name, else the base. */
static shards *olm_ov_src(Model *m, const char *name) {
    OlmOverlay *ov = g_ov;
    if (ov && st_find(&ov->S, name)) return &ov->S;
    return &m->S;
}

static void olm_ov_free(OlmOverlay *ov) {
    if (!ov) return;
    if (ov->kn) {
        for (int l = 0; l < ov->nkn_layers; l++) {
            free(ov->kn[l].w8); free(ov->kn[l].sc); free(ov->kn[l].gw); free(ov->kn[l].gb); free(ov->kn[l].slots);
        }
        free(ov->kn);
    }
    st_destroy(&ov->S);
    free(ov->cells);
    free(ov);
}

static int olm_ov_has(const OlmOverlay *ov, int layer, int eid) {
    if (!ov) return 0;
    for (int i = 0; i < ov->ncells; i++) if (ov->cells[2*i] == layer && ov->cells[2*i+1] == eid) return 1;
    return 0;
}

/* Open and validate an overlay directory. Returns NULL (with a reason on stderr) when anything
 * about it is wrong: an overlay that fails validation is never half-applied. */
static OlmOverlay *olm_ov_open(Model *m, const char *dir) {
    const Cfg *c = &m->c;
    char path[1200];
    snprintf(path, sizeof path, "%s/overlay.json", dir);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[OVERLAY] %s: no overlay.json\n", dir); return NULL; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n <= 0 || n > (8 << 20)) { fclose(f); fprintf(stderr, "[OVERLAY] overlay.json size %ld refused\n", n); return NULL; }
    char *buf = malloc((size_t)n + 1);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(buf); return NULL; }
    fclose(f); buf[n] = 0;
    char *arena = NULL; jval *j = json_parse(buf, &arena);
    free(buf);
    OlmOverlay *ov = NULL;
    if (!j) { fprintf(stderr, "[OVERLAY] overlay.json does not parse\n"); goto out; }
    jval *fmt = json_get(j, "format"), *fp = json_get(j, "base_fingerprint"),
         *ver = json_get(j, "version"), *cells = json_get(j, "cells"), *know = json_get(j, "knowledge");
    int v2 = fmt && fmt->t == J_STR && !strcmp(fmt->str, OLM_OV_FORMAT2);
    if (!fmt || fmt->t != J_STR || (strcmp(fmt->str, OLM_OV_FORMAT) && !v2)) {
        fprintf(stderr, "[OVERLAY] format is not %s or %s, refusing\n", OLM_OV_FORMAT, OLM_OV_FORMAT2); goto out; }
    if (!v2) know = NULL;
    if (know && (know->t != J_ARR || know->len > c->n_layers)) {
        fprintf(stderr, "[OVERLAY] knowledge is not [[layer, K], ...], refusing\n"); goto out; }
    char want[32]; snprintf(want, sizeof want, "%016llx", olm_ov_fingerprint(c));
    if (!fp || fp->t != J_STR || strcmp(fp->str, want)) {
        fprintf(stderr, "[OVERLAY] base_fingerprint %s does not match this model (%s), refusing\n",
                fp && fp->t == J_STR ? fp->str : "?", want); goto out; }
    if (!cells || cells->t != J_ARR || cells->len > c->n_layers * c->n_experts ||
        (cells->len < 1 && !(know && know->len > 0))) {
        fprintf(stderr, "[OVERLAY] cells missing or out of range, refusing\n"); goto out; }
    snprintf(path, sizeof path, "%s/overlay.safetensors", dir);
    if (access(path, R_OK)) { fprintf(stderr, "[OVERLAY] %s missing\n", path); goto out; }
    ov = calloc(1, sizeof *ov);
    ov->cells = malloc(sizeof(int) * 2 * (size_t)(cells->len ? cells->len : 1));
    ov->ncells = cells->len;
    ov->version = ver && ver->t == J_NUM ? (long long)ver->num : 0;
    snprintf(ov->dir, sizeof ov->dir, "%s", dir);
    for (int i = 0; i < cells->len; i++) {
        jval *p = cells->kids[i];
        if (!p || p->t != J_ARR || p->len != 2 || p->kids[0]->t != J_NUM || p->kids[1]->t != J_NUM) {
            fprintf(stderr, "[OVERLAY] cells[%d] is not [layer, expert]\n", i); free(ov->cells); free(ov); ov = NULL; goto out; }
        int L = (int)p->kids[0]->num, E = (int)p->kids[1]->num;
        if (L < 0 || L >= c->n_layers || E < 0 || E >= c->n_experts) {
            fprintf(stderr, "[OVERLAY] cell [%d,%d] outside this model, refusing\n", L, E); free(ov->cells); free(ov); ov = NULL; goto out; }
        ov->cells[2*i] = L; ov->cells[2*i+1] = E;
    }
    if (know && know->len > 0) {
        ov->kn = calloc((size_t)c->n_layers, sizeof(OlmKn));
        ov->nkn_layers = c->n_layers;
        for (int i = 0; i < know->len; i++) {
            jval *p = know->kids[i];
            if (!p || p->t != J_ARR || p->len != 2 || p->kids[0]->t != J_NUM || p->kids[1]->t != J_NUM) {
                fprintf(stderr, "[OVERLAY] knowledge[%d] is not [layer, K]\n", i); olm_ov_free(ov); ov = NULL; goto out; }
            int L = (int)p->kids[0]->num, K = (int)p->kids[1]->num;
            if (L < 0 || L >= c->n_layers || K < 1 || K > 64 || ov->kn[L].K) {
                fprintf(stderr, "[OVERLAY] knowledge [%d,%d] out of range or repeated, refusing\n", L, K); olm_ov_free(ov); ov = NULL; goto out; }
            ov->kn[L].K = K; ov->nkn += K;
        }
    }
    st_init(&ov->S, dir);
    /* every tensor must be an expert of a listed cell (or a declared knowledge expert / gate)
     * with the base container's exact size */
    const long long want_w = 3LL * c->inter * c->hidden, want_s = 2LL * c->inter + c->hidden;
    int ok = 1;
    for (int i = 0; ok && i < ov->S.n; i++) {
        const st_tensor *t = &ov->S.t[i];
        int L, E; char kind[32];
        if (ov->kn && sscanf(t->name, "model.layers.%d.mlp.knowledge.%31s", &L, kind) == 2) {
            int J; char sub[32];
            if (L < 0 || L >= c->n_layers || !ov->kn[L].K) ok = 0;
            else if (!strcmp(kind, "gate_w")) ok = t->dtype == 2 && t->numel == (long long)ov->kn[L].K * c->hidden;
            else if (!strcmp(kind, "gate_b")) ok = t->dtype == 2 && t->numel == ov->kn[L].K;
            else if (sscanf(kind, "%d.%31s", &J, sub) == 2 && J >= 0 && J < ov->kn[L].K) {
                if (!strcmp(sub, "merged_weight")) ok = t->nbytes == want_w;
                else if (!strcmp(sub, "qs")) ok = t->numel == want_s;
                else ok = 0;
            } else ok = 0;
            if (!ok) fprintf(stderr, "[OVERLAY] knowledge tensor %s has the wrong size, kind or layer, refusing\n", t->name);
            continue;
        }
        if (sscanf(t->name, "model.layers.%d.mlp.experts.%d.%31s", &L, &E, kind) != 3 || !olm_ov_has(ov, L, E)) {
            fprintf(stderr, "[OVERLAY] tensor %s is not an expert of a listed cell, refusing\n", t->name); ok = 0; break; }
        if (!strcmp(kind, "merged_weight")) ok = t->nbytes == want_w;
        else if (!strcmp(kind, "qs")) ok = t->numel == want_s;
        else ok = 0;
        if (!ok) fprintf(stderr, "[OVERLAY] tensor %s has the wrong size or kind, refusing\n", t->name);
    }
    for (int i = 0; ok && i < ov->ncells; i++) {
        char a[160], b[160];
        snprintf(a, sizeof a, "model.layers.%d.mlp.experts.%d.merged_weight", ov->cells[2*i], ov->cells[2*i+1]);
        snprintf(b, sizeof b, "model.layers.%d.mlp.experts.%d.qs", ov->cells[2*i], ov->cells[2*i+1]);
        if (!st_find(&ov->S, a) || !st_find(&ov->S, b)) {
            fprintf(stderr, "[OVERLAY] cell [%d,%d] lacks merged_weight or qs, refusing\n", ov->cells[2*i], ov->cells[2*i+1]); ok = 0; }
    }
    for (int L = 0; ok && ov->kn && L < c->n_layers; L++) {      /* load knowledge experts into RAM */
        OlmKn *k = &ov->kn[L];
        if (!k->K) continue;
        char nm[200];
        k->w8 = malloc((size_t)k->K * want_w);
        k->sc = malloc(sizeof(float) * (size_t)k->K * want_s);
        k->gw = malloc(sizeof(float) * (size_t)k->K * c->hidden);
        k->gb = malloc(sizeof(float) * (size_t)k->K);
        k->slots = calloc((size_t)k->K, sizeof(Slot));
        if (!k->w8 || !k->sc || !k->gw || !k->gb || !k->slots) { fprintf(stderr, "[OVERLAY] out of memory\n"); ok = 0; break; }
        snprintf(nm, sizeof nm, "model.layers.%d.mlp.knowledge.gate_w", L);
        if (!st_find(&ov->S, nm)) { fprintf(stderr, "[OVERLAY] %s missing\n", nm); ok = 0; break; }
        st_read_f32(&ov->S, nm, k->gw, 0);
        snprintf(nm, sizeof nm, "model.layers.%d.mlp.knowledge.gate_b", L);
        if (!st_find(&ov->S, nm)) { fprintf(stderr, "[OVERLAY] %s missing\n", nm); ok = 0; break; }
        st_read_f32(&ov->S, nm, k->gb, 0);
        for (int J = 0; J < k->K; J++) {
            int8_t *w = k->w8 + (size_t)J * want_w;
            float *sc = k->sc + (size_t)J * want_s;
            snprintf(nm, sizeof nm, "model.layers.%d.mlp.knowledge.%d.merged_weight", L, J);
            if (!st_find(&ov->S, nm)) { fprintf(stderr, "[OVERLAY] %s missing\n", nm); ok = 0; break; }
            st_read_raw(&ov->S, nm, w, 0);
            snprintf(nm, sizeof nm, "model.layers.%d.mlp.knowledge.%d.qs", L, J);
            if (!st_find(&ov->S, nm)) { fprintf(stderr, "[OVERLAY] %s missing\n", nm); ok = 0; break; }
            st_read_f32(&ov->S, nm, sc, 0);
            cps_scale(sc, c->inter, c->hidden, g_cps_want);   /* the strength (coli_packs_status.h) */
            Slot *sl = &k->slots[J];
            const size_t ID = (size_t)c->inter * c->hidden;
            sl->eid = -1; sl->g = w; sl->u = w + ID; sl->d = w + 2 * ID;
            sl->gs = sc; sl->us = sc + c->inter; sl->ds = sc + 2 * c->inter;
        }
    }
    if (!ok) { olm_ov_free(ov); ov = NULL; }
out:
    free(arena);
    return ov;
}

static void load_expert_merged(Model *m, int layer, int eid, Slot *s);
static void moe_expert_row(const Model *m, const Slot *e, const float *xs, float *g, float *u, float *hh);

/* B-slot: out[s] += sigmoid(gate_w[j].x[s] + gate_b[j]) * knowledge_expert_j(x[s]).
 * Runs on the forward thread; g_ov only changes in olm_overlay_poll at the start of a forward
 * pass on this same thread, so the buffers cannot be freed underneath it. */
static void olm_kn_apply(Model *m, int layer, const float *x, int S, float *out) {
    OlmOverlay *ov = g_ov;
    if (!ov || !ov->kn || !ov->kn[layer].K) return;
    const OlmKn *k = &ov->kn[layer];
    const int D = m->c.hidden, I = m->c.inter;
    float *g = malloc(sizeof(float) * (size_t)I), *u = malloc(sizeof(float) * (size_t)I),
          *hh = malloc(sizeof(float) * (size_t)D);
    for (int s = 0; s < S; s++) {
        const float *xs = x + (int64_t)s * D;
        float *os = out + (int64_t)s * D;
        for (int j = 0; j < k->K; j++) {
            const float *w = k->gw + (int64_t)j * D;
            double z = k->gb[j];
            for (int d = 0; d < D; d++) z += (double)w[d] * xs[d];
            const float gate = (float)(1.0 / (1.0 + exp(-z)));
            moe_expert_row(m, &k->slots[j], xs, g, u, hh);
            for (int d = 0; d < D; d++) os[d] += gate * hh[d];
        }
    }
    free(g); free(u); free(hh);
}

/* Swap the active overlay (dir NULL or "" deactivates). Returns 0 on success; on any failure the
 * previous overlay stays active. Call between forward passes, never while holding g_pilot_mx. */
static int olm_overlay_set_inner(Model *m, const char *dir);
/* The swap, then the status file; a successful swap drops reusable prompt state (built under the
 * old overlay) so a resent transcript is computed again. */
static int olm_overlay_set(Model *m, const char *dir) {
    int r = olm_overlay_set_inner(m, dir);
    if (!r) kv_prefix_clear(&m->kvp);
    const OlmOverlay *ov = g_ov;
    cps_write(g_ov_epoch, ov ? ov->dir : NULL, ov ? ov->version : 0, ov ? ov->nkn : 0, ov ? ov->ncells : 0,
              r ? "refused" : "ok");
    return r;
}
static int olm_overlay_set_inner(Model *m, const char *dir) {
    if (m->gx) { fprintf(stderr, "[OVERLAY] a GGUF model: overlays need the merged int8 container\n"); return -1; }
#ifdef COLI_VULKAN
    /* The GPU may run the dense part (attention, router, head) while the experts stay on the CPU:
     * COLI_VULKAN=1 COLI_VK_TIER=0 COLI_VK_CHAIN=0. Then every expert is read from the RAM slots this
     * overlay replaces, and knowledge experts run in moe(): allowed. The expert tier and the layer
     * chain keep their own copies of expert weights on the device: refused, never half-applied. */
    if (g_vk_ready && (vkt_ready() || g_vk_chain)) {
        fprintf(stderr, "[OVERLAY] refused: the Vulkan %s keeps its own expert copies on the device. "
                "Use COLI_VK_TIER=0 COLI_VK_CHAIN=0 (dense on the GPU, experts on the CPU)\n",
                g_vk_chain ? "layer chain" : "expert tier");
        return -1;
    }
#endif
    OlmOverlay *nw = NULL;
    if (dir && *dir) { nw = olm_ov_open(m, dir); if (!nw) return -1; }
    if (g_ov_dev_check && !g_ov_dev_check(m, nw)) { olm_ov_free(nw); return -1; }
    const Cfg *c = &m->c;
    pthread_mutex_lock(&g_pilot_mx);
    OlmOverlay *old = g_ov;
    g_ov_waiting = 1;
    for (;;) {                                   /* 1. drain loaders and affected readers */
        int busy = 0;
        for (int l = 0; l < c->n_layers && !busy; l++) {
            LCache *lc = &m->cache[l];
            for (int i = 0; i < lc->n; i++) {
                Slot *s = &lc->slots[i];
                if (s->eid < -1) { busy = 1; break; }          /* a load in flight */
                if (s->busy && s->eid >= 0 && (olm_ov_has(old, l, s->eid) || olm_ov_has(nw, l, s->eid))) { busy = 1; break; }
            }
        }
        if (!busy) break;
        pthread_cond_wait(&g_pilot_cv, &g_pilot_mx);
    }
    g_ov = nw;                                   /* 2. publish */
    g_ov_epoch++;
    int reloaded = 0;                            /* 3. re-read resident affected slots */
    for (int pass = 0; pass < 2; pass++) {
        const OlmOverlay *src = pass ? nw : old;
        for (int i = 0; src && i < src->ncells; i++) {
            int l = src->cells[2*i], e = src->cells[2*i+1];
            if (pass && olm_ov_has(old, l, e)) continue;       /* already done in pass 0 */
            Slot *s = slot_indexed(m, l, e);
            if (s) { load_expert_merged(m, l, e, s); reloaded++; }
        }
    }
    g_ov_waiting = 0;
    pthread_cond_broadcast(&g_pilot_cv);
    pthread_mutex_unlock(&g_pilot_mx);
    if (g_ov_dev_sync) g_ov_dev_sync(m, old, nw);   /* the card's copies of the swapped experts */
    olm_ov_free(old);
    g_cps_strength = nw ? g_cps_want : 1.f;
    if (nw) fprintf(stderr, "[OVERLAY] active: %s version %lld, %d expert(s), %d knowledge expert(s), %d resident slot(s) re-read, epoch %llu%s\n",
                    nw->dir, nw->version, nw->ncells, nw->nkn, reloaded, g_ov_epoch, cps_sfx());
    else    fprintf(stderr, "[OVERLAY] off: base experts only, %d resident slot(s) re-read, epoch %llu\n", reloaded, g_ov_epoch);
    return 0;
}

static void olm_ov_on_hup(int sig) { (void)sig; g_ov_reload = 1; }

/* Start-up: OLM_OVERLAY activates an overlay; OLM_OVERLAY_CTL arms the SIGHUP live swap. */
static void olm_overlay_from_env(Model *m) {
    cps_init("olmoe", olm_ov_fingerprint(&m->c));   /* the control file, env or /tmp/colibri-packs/<pid>.ctl */
    cps_write(g_ov_epoch, NULL, 0, 0, 0, "ok");
    const char *d = getenv("OLM_OVERLAY");
    if (!d || !*d) d = getenv("COLI_OVERLAY");
    const char *ps = getenv("COLI_PACK_STRENGTH");
    if (d && *d && ps && *ps) {
        char buf[1100]; snprintf(buf, sizeof buf, "x\t%s", ps);
        if (!cps_split(buf, &g_cps_want)) { fprintf(stderr, "[OVERLAY] COLI_PACK_STRENGTH=%s refused, exiting\n", ps); exit(1); }
    }
    if (d && *d && olm_overlay_set(m, d)) { fprintf(stderr, "[OVERLAY] OLM_OVERLAY=%s refused, exiting\n", d); exit(1); }
    signal(SIGHUP, olm_ov_on_hup);
}

/* Called at the start of every forward pass: applies a requested live swap. */
static void olm_overlay_poll(Model *m) {
    if (!g_ov_reload) return;
    g_ov_reload = 0;
    const char *ctl = g_cps_ctl;
    if (!*ctl) return;
    char dir[1024] = {0};
    FILE *f = fopen(ctl, "rb");
    if (!f) { fprintf(stderr, "[OVERLAY] OLM_OVERLAY_CTL %s unreadable, overlay unchanged\n", ctl); return; }
    if (!fgets(dir, sizeof dir, f)) dir[0] = 0;
    fclose(f);
    dir[strcspn(dir, "\r\n")] = 0;
    float st;
    if (!cps_split(dir, &st)) {               /* a bad strength changes nothing */
        const OlmOverlay *ov = g_ov;
        cps_write(g_ov_epoch, ov ? ov->dir : NULL, ov ? ov->version : 0, ov ? ov->nkn : 0, ov ? ov->ncells : 0, "refused");
        return;
    }
    g_cps_want = st;
    olm_overlay_set(m, dir);
}
#endif
