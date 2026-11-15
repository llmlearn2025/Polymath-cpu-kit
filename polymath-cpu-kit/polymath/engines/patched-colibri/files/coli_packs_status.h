/* coli_packs_status.h — how a client (the Hermes plugin) finds and checks a colibri engine that
 * takes knowledge packs (moe-knowledge). Shared by olmoe_overlay.h and coli_kn.h.
 *
 * Every pack-capable engine, at start:
 *   - takes its control file from OLM_OVERLAY_CTL / COLI_OVERLAY_CTL, or uses
 *     /tmp/colibri-packs/<pid>.ctl, and arms SIGHUP (a live swap reads that file);
 *   - writes /tmp/colibri-packs/<pid>.json (COLI_PACKS_STATUS overrides the path):
 *       {"packs": 1, "pid", "ppid", "engine", "snap", "fingerprint", "ctl", "epoch", "active",
 *        "version", "knowledge_experts", "modified_experts", "last", "time"}
 *     and rewrites it after every swap attempt ("last": "ok" or "refused").
 * The control file holds one line: the pack directory (empty = no pack), optionally followed by a TAB and a
 * strength (0 < strength <= 4, default 1; COLI_PACK_STRENGTH at start). The strength multiplies the knowledge
 * experts' output: it is folded into their down-projection row scales when the pack is read, so the CPU path
 * and every device copy compute it at no cost. Modified routed experts (method A) are not scaled.
 * A process without that file is not a pack-capable colibri engine: clients refuse to swap. */
#ifndef COLI_PACKS_STATUS_H
#define COLI_PACKS_STATUS_H
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define COLI_PACKS_DIR "/tmp/colibri-packs"

static char g_cps_ctl[1024], g_cps_status[1024];
static const char *g_cps_engine = "?", *g_cps_snap = "";
static char g_cps_fp[32];
static float g_cps_strength = 1.f;   /* the active pack's strength */
static float g_cps_want = 1.f;       /* the strength of the pack being opened */

/* "dir<TAB>strength" -> dir (in place) and the requested strength; 0 if the strength is not a number in (0, 4] */
static int cps_split(char *dir, float *st) {
    *st = 1.f;
    char *tab = strchr(dir, '\t');
    if (!tab) return 1;
    *tab = 0;
    char *end = NULL;
    double v = strtod(tab + 1, &end);
    while (end && (*end == ' ' || *end == '\t')) end++;
    if (!end || end == tab + 1 || *end || !(v > 0.0) || v > 4.0) {
        fprintf(stderr, "[OVERLAY] strength %s refused: it must be a number above 0 and at most 4\n", tab + 1); return 0; }
    *st = (float)v;
    return 1;
}
/* ", strength xS" for the swap log line, empty at strength 1 */
static const char *cps_sfx(void) {
    static char b[48];
    if (g_cps_strength == 1.f) b[0] = 0; else snprintf(b, sizeof b, ", strength x%g", g_cps_strength);
    return b;
}
/* fold the strength into one knowledge expert's row scales [I | I | D]: the down rows only */
static void cps_scale(float *sc, int I, int D, float st) {
    if (st == 1.f) return;
    for (int d = 0; d < D; d++) sc[2 * (size_t)I + d] *= st;
}

static int g_cps_own_ctl;
static void cps_cleanup(void) {   /* a stopped engine leaves no status behind */
    if (g_cps_status[0]) unlink(g_cps_status);
    if (g_cps_own_ctl) unlink(g_cps_ctl);
}
/* the control file (env or default) and the status path; creates the default directory */
static const char *cps_init(const char *engine, unsigned long long fingerprint) {
    g_cps_engine = engine;
    const char *s = getenv("SNAP");
    g_cps_snap = s ? s : "";
    snprintf(g_cps_fp, sizeof g_cps_fp, "%016llx", fingerprint);
    const char *c = getenv("COLI_OVERLAY_CTL");
    if (!c || !*c) c = getenv("OLM_OVERLAY_CTL");
    mkdir(COLI_PACKS_DIR, 0700);
    if (c && *c) snprintf(g_cps_ctl, sizeof g_cps_ctl, "%s", c);
    else {
        snprintf(g_cps_ctl, sizeof g_cps_ctl, "%s/%d.ctl", COLI_PACKS_DIR, (int)getpid());
        FILE *f = fopen(g_cps_ctl, "wb"); if (f) { fputc('\n', f); fclose(f); }
        g_cps_own_ctl = 1;
    }
    const char *st = getenv("COLI_PACKS_STATUS");
    if (st && *st) snprintf(g_cps_status, sizeof g_cps_status, "%s", st);
    else snprintf(g_cps_status, sizeof g_cps_status, "%s/%d.json", COLI_PACKS_DIR, (int)getpid());
    atexit(cps_cleanup);
    return g_cps_ctl;
}

static void cps_json_str(FILE *f, const char *s) {
    fputc('"', f);
    for (; s && *s; s++) {
        if (*s == '"' || *s == '\\') { fputc('\\', f); fputc(*s, f); }
        else if ((unsigned char)*s < 0x20) fprintf(f, "\\u%04x", *s);
        else fputc(*s, f);
    }
    fputc('"', f);
}

/* the status file, written whole and renamed into place */
static void cps_write(unsigned long long epoch, const char *active, long long version, int nkn, int ncells, const char *last) {
    if (!g_cps_status[0]) return;
    char tmp[1100];
    snprintf(tmp, sizeof tmp, "%s.tmp", g_cps_status);
    FILE *f = fopen(tmp, "wb");
    if (!f) return;
    fprintf(f, "{\"packs\": 1, \"pid\": %d, \"ppid\": %d, \"engine\": ", (int)getpid(), (int)getppid());
    cps_json_str(f, g_cps_engine); fputs(", \"snap\": ", f); cps_json_str(f, g_cps_snap);
    fputs(", \"fingerprint\": ", f); cps_json_str(f, g_cps_fp);
    fputs(", \"ctl\": ", f); cps_json_str(f, g_cps_ctl);
    fprintf(f, ", \"epoch\": %llu, \"active\": ", epoch);
    if (active && *active) cps_json_str(f, active); else fputs("null", f);
    fprintf(f, ", \"version\": %lld, \"knowledge_experts\": %d, \"modified_experts\": %d, \"last\": ", version, nkn, ncells);
    cps_json_str(f, last);
    fprintf(f, ", \"strength\": %g", active && *active ? g_cps_strength : 1.0);
    static unsigned long long attempts;
    fprintf(f, ", \"attempts\": %llu, \"time\": %lld}\n", attempts++, (long long)time(NULL));
    fclose(f);
    rename(tmp, g_cps_status);
}
#endif
