/* Overlay swap stress: worker threads read random experts through the real cache while the main
 * thread swaps overlays (none / A / B) over and over. Every expert a worker is handed must hold
 * exactly the bytes of the source the overlay resolution names at that moment, and after the
 * final deactivation every resident slot must hold the base bytes.
 *
 *   SNAP=<converted tiny OLMoE> OV_A=<overlay dir> OV_B=<overlay dir> ./tests/test_olmoe_overlay_swap [swaps] [cap]
 */
#define main olmoe_main_unused
#include "../olmoe.c"
#undef main

static Model g_m;
static volatile int g_ov_test_stop;
static long g_checks, g_bad;
static pthread_mutex_t g_cnt = PTHREAD_MUTEX_INITIALIZER;

static int check_slot(Model *m, int l, int e, const Slot *s, int8_t *wbuf, float *sbuf) {
    const Cfg *c = &m->c;
    const long long nw = 3LL * c->inter * c->hidden, ns = 2LL * c->inter + c->hidden;
    char nm[160], qn[160];
    snprintf(nm, sizeof nm, "model.layers.%d.mlp.experts.%d.merged_weight", l, e);
    snprintf(qn, sizeof qn, "model.layers.%d.mlp.experts.%d.qs", l, e);
    pthread_mutex_lock(&g_pilot_mx);
    shards *src = olm_ov_src(m, nm);
    st_read_raw(src, nm, wbuf, 0);
    st_read_f32(src, qn, sbuf, 0);
    int ok = !memcmp(s->g, wbuf, (size_t)nw) && !memcmp(s->gs, sbuf, (size_t)ns * sizeof(float));
    pthread_mutex_unlock(&g_pilot_mx);
    return ok;
}

static void *worker(void *arg) {
    unsigned seed = (unsigned)(uintptr_t)arg;
    const Cfg *c = &g_m.c;
    int8_t *wbuf = malloc((size_t)3 * c->inter * c->hidden);
    float *sbuf = malloc(sizeof(float) * (size_t)(2 * c->inter + c->hidden));
    while (!g_ov_test_stop) {
        int l = rand_r(&seed) % c->n_layers, e = rand_r(&seed) % c->n_experts;
        Slot *s = NULL;
        expert_get(&g_m, l, e, &s);
        int ok = check_slot(&g_m, l, e, s, wbuf, sbuf);
        expert_put(s);
        pthread_mutex_lock(&g_cnt);
        g_checks++;
        if (!ok) { g_bad++; if (g_bad < 5) fprintf(stderr, "FAIL: layer %d expert %d holds stale bytes\n", l, e); }
        pthread_mutex_unlock(&g_cnt);
    }
    free(wbuf); free(sbuf);
    return NULL;
}

int main(int argc, char **argv) {
    const char *snap = getenv("SNAP"), *a = getenv("OV_A"), *b = getenv("OV_B");
    if (!snap || !a || !b) { fprintf(stderr, "need SNAP, OV_A, OV_B\n"); return 2; }
    int swaps = argc > 1 ? atoi(argv[1]) : 2000, cap = argc > 2 ? atoi(argv[2]) : 3;
    model_init(&g_m, snap, cap, 8);
    pthread_t th[3];
    for (int i = 0; i < 3; i++) pthread_create(&th[i], NULL, worker, (void *)(uintptr_t)(i + 1));
    const char *seq[3] = {a, NULL, b};
    int fails = 0;
    for (int i = 0; i < swaps; i++) {
        if (olm_overlay_set(&g_m, seq[i % 3])) fails++;
        if (i % 500 == 0) fprintf(stderr, "swap %d, checks so far %ld\n", i, g_checks);
    }
    g_ov_test_stop = 1;
    for (int i = 0; i < 3; i++) pthread_join(th[i], NULL);
    olm_overlay_set(&g_m, NULL);
    /* every resident slot must now hold base bytes */
    const Cfg *c = &g_m.c;
    int8_t *wbuf = malloc((size_t)3 * c->inter * c->hidden);
    float *sbuf = malloc(sizeof(float) * (size_t)(2 * c->inter + c->hidden));
    int resident = 0, stale = 0;
    for (int l = 0; l < c->n_layers; l++)
        for (int i = 0; i < g_m.cache[l].n; i++) {
            Slot *s = &g_m.cache[l].slots[i];
            if (s->eid < 0) continue;
            resident++;
            if (!check_slot(&g_m, l, s->eid, s, wbuf, sbuf)) stale++;
        }
    printf("swaps=%d swap_failures=%d checks=%ld stale_reads=%ld resident_after_off=%d stale_after_off=%d\n",
           swaps, fails, g_checks, g_bad, resident, stale);
    int ok = !fails && !g_bad && !stale && g_checks > 0;
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
