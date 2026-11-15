/* WINDOWS VERSION (see windows/README.md): the live swap is noticed by the control file changing, not by SIGHUP.
 * coli_kn.h — knowledge experts ("law packs", B-slot) for a colibri MoE engine (moe-knowledge).
 *
 * A pack adds K experts per chosen layer beside the routed ones, each with a sigmoid gate, and
 * every token gets
 *     out += sigmoid(gate_w[j] . x + gate_b[j]) * expert_j(x)          x: the MoE block's input
 * on top of the routed experts' sum. The router, the routed experts and the expert cache are
 * untouched, so a model without a pack, or with the pack removed, is exactly the base model.
 *
 * Engine-neutral: it needs only st.h, json.h and the model's geometry. An engine adds
 *   - coli_kn_apply(layer, x, S, out)   after its routed experts on the CPU path;
 *   - coli_kn_poll(m)                   at the start of a forward pass (applies a live swap);
 *   - coli_kn_from_env(m, geometry)     once the model (and any device chain) is up;
 *   - optionally the device hooks g_kn_dev_prepare / g_kn_dev_commit (a GPU chain's copy of the
 *     pack) and g_kn_invalidate (drop cached prompt states built under the old pack).
 *
 * Pack directory (moe-knowledge export/to_colibri_overlay.py, format colibri-olmoe-overlay-v2):
 *   overlay.safetensors  model.layers.L.mlp.knowledge.J.merged_weight  I8 [gate I*D | up I*D | down D*I]
 *                        model.layers.L.mlp.knowledge.J.qs             F32 row scales [I | I | D]
 *                        model.layers.L.mlp.knowledge.gate_w / gate_b  F32 [K, D] / [K]
 *   overlay.json         {"format", "version", "base_fingerprint", "cells": [], "knowledge": [[L, K], ...]}
 * Packs that also modify routed experts ("cells", method A) are refused here: that needs the
 * engine's expert loader (olmoe_overlay.h does it for OLMoE).
 *
 * Activation: COLI_OVERLAY=<dir> (or OLM_OVERLAY) at start; a live swap re-reads the file named
 * by COLI_OVERLAY_CTL (or OLM_OVERLAY_CTL; a directory, or empty to remove the pack) after
 * kill -HUP, at the start of the next forward pass. A pack that fails any check is never
 * half-applied: the previous one stays. */
#ifndef COLI_KN_H
#define COLI_KN_H
#include <signal.h>
#include <math.h>
#include "coli_packs_status.h"   /* the status file a client (the Hermes plugin) checks */

#define COLI_KN_FORMAT "colibri-olmoe-overlay-v2"

typedef struct { int K; int8_t *w8; float *sc, *gw, *gb; } ColiKnLayer;
typedef struct { char dir[1024]; long long version; int L, D, I, nkn; ColiKnLayer *kn; } ColiKn;
typedef struct { int L, D, I, E, topk, vocab; } ColiKnGeom;

static ColiKn *g_kn = NULL;
static ColiKnGeom g_kn_geom;
static unsigned long long g_kn_epoch = 0;
static volatile sig_atomic_t g_kn_reload = 0;
static const char *g_kn_refuse = NULL;                     /* the engine runs a path packs cannot use */
static int  (*g_kn_dev_prepare)(void *m, const ColiKn *nw) = NULL;   /* device copy of nw; 0 refuses the swap */
static void (*g_kn_dev_commit)(void *m, const ColiKn *nw) = NULL;    /* the prepared copy current, the old one freed */
static void (*g_kn_invalidate)(void *m) = NULL;            /* cached prompt states are stale */

static unsigned long long coli_kn_fingerprint(const ColiKnGeom *g) {   /* = export/to_colibri_overlay.py */
    unsigned long long h = 1469598103934665603ULL;
    long long v[6] = {g->D, g->I, g->L, g->E, g->topk, g->vocab};
    for (int k = 0; k < 6; k++)
        for (int i = 0; i < 8; i++) { h ^= (unsigned long long)((v[k] >> (8 * i)) & 0xff); h *= 1099511628211ULL; }
    return h;
}

static void coli_kn_free(ColiKn *k) {
    if (!k) return;
    if (k->kn) for (int l = 0; l < k->L; l++) { free(k->kn[l].w8); free(k->kn[l].sc); free(k->kn[l].gw); free(k->kn[l].gb); }
    free(k->kn); free(k);
}

/* Read and check a pack directory into RAM. NULL (with a reason) when anything is wrong. */
static ColiKn *coli_kn_open(const char *dir) {
    const ColiKnGeom *g = &g_kn_geom;
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
    ColiKn *k = NULL;
    shards S; int s_open = 0;
    if (!j) { fprintf(stderr, "[OVERLAY] overlay.json does not parse\n"); goto out; }
    jval *fmt = json_get(j, "format"), *fp = json_get(j, "base_fingerprint"), *ver = json_get(j, "version"),
         *cells = json_get(j, "cells"), *know = json_get(j, "knowledge");
    if (!fmt || fmt->t != J_STR || strcmp(fmt->str, COLI_KN_FORMAT)) {
        fprintf(stderr, "[OVERLAY] format is not %s (a knowledge pack), refusing\n", COLI_KN_FORMAT); goto out; }
    char want[32]; snprintf(want, sizeof want, "%016llx", coli_kn_fingerprint(g));
    if (!fp || fp->t != J_STR || strcmp(fp->str, want)) {
        fprintf(stderr, "[OVERLAY] base_fingerprint %s does not match this model (%s), refusing\n",
                fp && fp->t == J_STR ? fp->str : "?", want); goto out; }
    if (cells && cells->t == J_ARR && cells->len > 0) {
        fprintf(stderr, "[OVERLAY] refused: the pack modifies routed experts (method A); this engine takes "
                        "knowledge experts only\n"); goto out; }
    if (!know || know->t != J_ARR || know->len < 1 || know->len > g->L) {
        fprintf(stderr, "[OVERLAY] knowledge is not [[layer, K], ...], refusing\n"); goto out; }
    k = calloc(1, sizeof *k);
    k->L = g->L; k->D = g->D; k->I = g->I;
    k->kn = calloc((size_t)g->L, sizeof(ColiKnLayer));
    k->version = ver && ver->t == J_NUM ? (long long)ver->num : 0;
    snprintf(k->dir, sizeof k->dir, "%s", dir);
    int ok = 1;
    for (int i = 0; ok && i < know->len; i++) {
        jval *p = know->kids[i];
        if (!p || p->t != J_ARR || p->len != 2 || p->kids[0]->t != J_NUM || p->kids[1]->t != J_NUM) {
            fprintf(stderr, "[OVERLAY] knowledge[%d] is not [layer, K]\n", i); ok = 0; break; }
        int L = (int)p->kids[0]->num, K = (int)p->kids[1]->num;
        if (L < 0 || L >= g->L || K < 1 || K > 64 || k->kn[L].K) {
            fprintf(stderr, "[OVERLAY] knowledge [%d,%d] out of range or repeated, refusing\n", L, K); ok = 0; break; }
        k->kn[L].K = K; k->nkn += K;
    }
    snprintf(path, sizeof path, "%s/overlay.safetensors", dir);
    if (ok && access(path, R_OK)) { fprintf(stderr, "[OVERLAY] %s missing\n", path); ok = 0; }
    if (ok) { st_init(&S, dir); s_open = 1; }
    const long long nw = 3LL * g->I * g->D, ns = 2LL * g->I + g->D;
    for (int i = 0; ok && i < S.n; i++) {          /* every tensor a declared knowledge tensor of the right size */
        const st_tensor *t = &S.t[i];
        int L, J; char kind[32], sub[32];
        if (sscanf(t->name, "model.layers.%d.mlp.knowledge.%31s", &L, kind) != 2 || L < 0 || L >= g->L || !k->kn[L].K) ok = 0;
        else if (!strcmp(kind, "gate_w")) ok = t->dtype == 2 && t->numel == (long long)k->kn[L].K * g->D;
        else if (!strcmp(kind, "gate_b")) ok = t->dtype == 2 && t->numel == k->kn[L].K;
        else if (sscanf(kind, "%d.%31s", &J, sub) == 2 && J >= 0 && J < k->kn[L].K)
            ok = !strcmp(sub, "merged_weight") ? t->nbytes == nw : !strcmp(sub, "qs") ? t->dtype == 2 && t->numel == ns : 0;
        else ok = 0;
        if (!ok) fprintf(stderr, "[OVERLAY] tensor %s has the wrong name, size or layer, refusing\n", t->name);
    }
    for (int L = 0; ok && L < g->L; L++) {         /* into RAM */
        ColiKnLayer *x = &k->kn[L];
        if (!x->K) continue;
        x->w8 = malloc((size_t)x->K * nw); x->sc = malloc(sizeof(float) * (size_t)x->K * ns);
        x->gw = malloc(sizeof(float) * (size_t)x->K * g->D); x->gb = malloc(sizeof(float) * (size_t)x->K);
        if (!x->w8 || !x->sc || !x->gw || !x->gb) { fprintf(stderr, "[OVERLAY] out of memory\n"); ok = 0; break; }
        char nm[200];
        snprintf(nm, sizeof nm, "model.layers.%d.mlp.knowledge.gate_w", L);
        if (!st_find(&S, nm)) { fprintf(stderr, "[OVERLAY] %s missing\n", nm); ok = 0; break; }
        st_read_f32(&S, nm, x->gw, 0);
        snprintf(nm, sizeof nm, "model.layers.%d.mlp.knowledge.gate_b", L);
        if (!st_find(&S, nm)) { fprintf(stderr, "[OVERLAY] %s missing\n", nm); ok = 0; break; }
        st_read_f32(&S, nm, x->gb, 0);
        for (int J = 0; ok && J < x->K; J++) {
            snprintf(nm, sizeof nm, "model.layers.%d.mlp.knowledge.%d.merged_weight", L, J);
            if (!st_find(&S, nm)) { fprintf(stderr, "[OVERLAY] %s missing\n", nm); ok = 0; break; }
            st_read_raw(&S, nm, x->w8 + (size_t)J * nw, 0);
            snprintf(nm, sizeof nm, "model.layers.%d.mlp.knowledge.%d.qs", L, J);
            if (!st_find(&S, nm)) { fprintf(stderr, "[OVERLAY] %s missing\n", nm); ok = 0; break; }
            st_read_f32(&S, nm, x->sc + (size_t)J * ns, 0);
            cps_scale(x->sc + (size_t)J * ns, g->I, g->D, g_cps_want);   /* the strength (coli_packs_status.h) */
        }
    }
    if (!ok) { coli_kn_free(k); k = NULL; }
out:
    if (s_open) st_destroy(&S);
    free(arena);
    return k;
}

/* y[o] = s[o] * sum_i W[o][i] x[i]: one int8 row-scaled matrix [O x I] */
static void coli_kn_mv(float *y, const int8_t *W, const float *s, const float *x, int O, int I) {
#pragma omp parallel for schedule(static) if (O >= 128)
    for (int o = 0; o < O; o++) {
        const int8_t *w = W + (size_t)o * I;
        float a = 0.f;
        for (int i = 0; i < I; i++) a += (float)w[i] * x[i];
        y[o] = a * s[o];
    }
}

/* out[s] += sum_j sigmoid(gate_w[j] . x[s] + gate_b[j]) * expert_j(x[s]) for layer `layer`
 * (x: S rows of D, the MoE input; out: the routed sum). Runs on the forward thread; g_kn only
 * changes in coli_kn_poll, on the same thread, between forward passes. */
static void coli_kn_apply(int layer, const float *x, int S, float *out) {
    const ColiKn *k = g_kn;
    if (!k || layer < 0 || layer >= k->L || !k->kn[layer].K) return;
    const ColiKnLayer *c = &k->kn[layer];
    const int D = k->D, I = k->I;
    const size_t nw = 3 * (size_t)I * D, ns = 2 * (size_t)I + D;
    float *g = malloc(sizeof(float) * (size_t)I), *u = malloc(sizeof(float) * (size_t)I), *y = malloc(sizeof(float) * (size_t)D);
    for (int s = 0; s < S; s++) {
        const float *xs = x + (size_t)s * D;
        float *os = out + (size_t)s * D;
        for (int j = 0; j < c->K; j++) {
            const float *w = c->gw + (size_t)j * D;
            double z = c->gb[j];
            for (int d = 0; d < D; d++) z += (double)w[d] * xs[d];
            const float gate = (float)(1.0 / (1.0 + exp(-z)));
            const int8_t *W = c->w8 + (size_t)j * nw; const float *sc = c->sc + (size_t)j * ns;
            coli_kn_mv(g, W, sc, xs, I, D);
            coli_kn_mv(u, W + (size_t)I * D, sc + I, xs, I, D);
            for (int i = 0; i < I; i++) g[i] = g[i] / (1.f + expf(-g[i])) * u[i];
            coli_kn_mv(y, W + 2 * (size_t)I * D, sc + 2 * I, g, D, I);
            for (int d = 0; d < D; d++) os[d] += gate * y[d];
        }
    }
    free(g); free(u); free(y);
}

/* Swap the active pack (dir NULL or "" removes it). 0 on success; on any failure the previous
 * pack stays active, on the CPU and on the device. Call between forward passes. */
static int coli_kn_set_inner(void *m, const char *dir);
static int coli_kn_set(void *m, const char *dir) {
    int r = coli_kn_set_inner(m, dir);
    const ColiKn *k = g_kn;
    cps_write(g_kn_epoch, k ? k->dir : NULL, k ? k->version : 0, k ? k->nkn : 0, 0, r ? "refused" : "ok");
    return r;
}
static int coli_kn_set_inner(void *m, const char *dir) {
    if (g_kn_refuse && dir && *dir) { fprintf(stderr, "[OVERLAY] refused: %s\n", g_kn_refuse); return -1; }
    ColiKn *nw = NULL;
    if (dir && *dir) { nw = coli_kn_open(dir); if (!nw) return -1; }
    double t0 = (double)clock() / CLOCKS_PER_SEC;
    if (g_kn_dev_prepare && !g_kn_dev_prepare(m, nw)) {
        fprintf(stderr, "[OVERLAY] the device could not take the pack; the previous one stays\n");
        coli_kn_free(nw); return -1;
    }
    ColiKn *old = g_kn;
    g_kn = nw;
    g_kn_epoch++;
    if (g_kn_dev_commit) g_kn_dev_commit(m, nw);
    if (g_kn_invalidate) g_kn_invalidate(m);
    coli_kn_free(old);
    (void)t0;
    g_cps_strength = nw ? g_cps_want : 1.f;
    if (nw) fprintf(stderr, "[OVERLAY] active: %s version %lld, 0 expert(s), %d knowledge expert(s), 0 resident slot(s) re-read, epoch %llu%s\n",
                    nw->dir, nw->version, nw->nkn, g_kn_epoch, cps_sfx());
    else    fprintf(stderr, "[OVERLAY] off: base experts only, 0 resident slot(s) re-read, epoch %llu\n", g_kn_epoch);
    return 0;
}

static const char *coli_kn_env(const char *a, const char *b) { const char *v = getenv(a); return v && *v ? v : getenv(b); }
static void coli_kn_on_hup(int sig) { (void)sig; g_kn_reload = 1; }

/* Start-up: the model's geometry, a pack named by COLI_OVERLAY, the SIGHUP live swap armed. */
static void coli_kn_from_env(void *m, const ColiKnGeom *geom, const char *engine_name) {
    g_kn_geom = *geom;
    cps_init(engine_name, coli_kn_fingerprint(geom));   /* control file: env or /tmp/colibri-packs/<pid>.ctl */
    cps_write(g_kn_epoch, NULL, 0, 0, 0, g_kn_refuse ? "packs refused on this path" : "ok");
    const char *d = coli_kn_env("COLI_OVERLAY", "OLM_OVERLAY");
    const char *ps = getenv("COLI_PACK_STRENGTH");
    if (d && *d && ps && *ps) {
        char buf[1100]; snprintf(buf, sizeof buf, "x\t%s", ps);
        if (!cps_split(buf, &g_cps_want)) { fprintf(stderr, "[OVERLAY] COLI_PACK_STRENGTH=%s refused, exiting\n", ps); exit(1); }
    }
    if (d && *d && coli_kn_set(m, d)) { fprintf(stderr, "[OVERLAY] %s refused, exiting\n", d); exit(1); }
    (void)coli_kn_on_hup;                     /* WINDOWS: no SIGHUP; coli_kn_poll reads the control file */
}

/* At the start of every forward pass: a requested live swap. */
static void coli_kn_poll(void *m) {
    if (!g_kn_reload && !(g_cps_ctl[0] && cps_ctl_changed(g_cps_ctl))) return;
    g_kn_reload = 0;
    const char *ctl = g_cps_ctl;
    if (!*ctl) return;
    char dir[1024] = {0};
    FILE *f = fopen(ctl, "rb");
    if (!f) { fprintf(stderr, "[OVERLAY] %s unreadable, pack unchanged\n", ctl); return; }
    if (!fgets(dir, sizeof dir, f)) dir[0] = 0;
    fclose(f);
    dir[strcspn(dir, "\r\n")] = 0;
    float st;
    if (!cps_split(dir, &st)) {               /* a bad strength changes nothing */
        const ColiKn *k = g_kn;
        cps_write(g_kn_epoch, k ? k->dir : NULL, k ? k->version : 0, k ? k->nkn : 0, 0, "refused");
        return;
    }
    g_cps_want = st;
    coli_kn_set(m, dir);
}
#endif
