#!/usr/bin/env bash
# Install the patched colibri engine: fetch upstream colibri at the pinned commit, apply Polymath's knowledge-pack
# patches, build the olmoe and qwen36 engines: a CPU build, a GPU (CUDA) build, or both side by side, so the
# engine can be switched in Packsmith (Settings → Engine → Engine build, or the Workbench) for testing.
#
#   ./install_patched_colibri.sh                 both builds when nvcc is there, else the CPU build only
#   ./install_patched_colibri.sh --both          CPU and GPU builds (needs the CUDA toolkit: installing it is up to you)
#   ./install_patched_colibri.sh --gpu           GPU build only
#   ./install_patched_colibri.sh --cpu           CPU build only (works everywhere, slower)
#   ./install_patched_colibri.sh --dir ~/colibri use / create the checkout here (an existing colibri checkout is fine)
#   ./install_patched_colibri.sh --in-venv       put colibri inside the Python venv folder (.venv/colibri)
#   ./install_patched_colibri.sh --arch portable CUDA architectures: native (this card, default) | portable | 89 | sm_120 ...
#   ./install_patched_colibri.sh --yes           no questions (Docker, CI)
#   ./install_patched_colibri.sh --rebuild       compile everything again (make -B), even what make thinks is current
#   ./install_patched_colibri.sh --no-polymath-c B-slot hooks only (by default the same binaries also get the polymath-c
#                                                hooks: Polymath packs with anchored routing, engines/polymath-c)
#
# Default location: $POLYMATH_HOME/engines/colibri (~/.polymath/engines/colibri). Safe to re-run: it resets only the
# four files the patches touch and rebuilds. Under sudo everything is created in YOUR home and owned by you.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/lib/common.sh
source "$HERE/scripts/lib/common.sh"
ENG="$HERE/engines/patched-colibri"
PMC="$HERE/engines/polymath-c"
WITH_PMC=1; [ -d "$PMC/patches" ] || WITH_PMC=0
UPSTREAM="${COLIBRI_UPSTREAM:-https://github.com/JustVugg/colibri}"
PIN="$(tr -d '[:space:]' < "$ENG/UPSTREAM_COMMIT")"
MODE=auto; DEST=""; IN_VENV=0; ARCH="${CUDA_ARCH:-native}"; RUN_TESTS=1; REBUILD=0
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) MODE=gpu ;; --cpu) MODE=cpu ;; --both) MODE=both ;;
    --dir) DEST="$2"; shift ;;
    --in-venv) IN_VENV=1 ;;
    --arch) ARCH="$2"; shift ;;
    --yes|-y) ASSUME_YES=1 ;;
    --no-tests) RUN_TESTS=0 ;;
    --no-polymath-c) WITH_PMC=0 ;;
    --rebuild) REBUILD=1 ;;
    -h|--help) sed -n 2,19p "$0"; exit 0 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac
  shift
done

REQ_MODE="$MODE"                                   # what was asked for (auto resolves below): part of the fingerprint
say "== Patched colibri (upstream $PIN)"
[ "$UNDER_SUDO" = 1 ] && echo "  running under sudo: installing for $REAL_USER in $REAL_HOME"
for t in git gcc make; do command -v "$t" >/dev/null || die "missing: $t   (sudo apt install git build-essential)"; done

# ---------------------------------------------------------------- GPU or CPU
NVCC="$(find_nvcc)"
if [ "$MODE" = auto ]; then
  if [ -n "$NVCC" ]; then MODE=both
  else
    MODE=cpu
    has_nvidia_driver && warn "NVIDIA driver found but no CUDA toolkit (nvcc): CPU build only"
    has_nvidia_driver || echo "  no NVIDIA driver or CUDA toolkit: CPU build only"
  fi
fi
BUILDS=(); case "$MODE" in both) BUILDS=(cpu gpu) ;; gpu) BUILDS=(gpu) ;; *) BUILDS=(cpu) ;; esac
case "$ARCH" in [0-9]*) ARCH="sm_$ARCH" ;; esac     # --arch 89 -> sm_89 (what nvcc -arch takes)
if [ "$MODE" != cpu ]; then
  [ -n "$NVCC" ] || { warn "--$MODE needs the CUDA toolkit (nvcc), which was not found"; cuda_help; exit 1; }
  export PATH="$(dirname "$NVCC"):$PATH"
  has_nvidia_driver || { [ "$ARCH" = native ] && { warn "no GPU visible: building for all supported cards (--arch portable)"; ARCH=portable; }; }
  ok "GPU build with $("$NVCC" --version | sed -n 's/.*release \([0-9.]*\).*/CUDA \1/p'), architectures: $ARCH"
fi
ok "builds: ${BUILDS[*]}"

# ---------------------------------------------------------------- where
if [ -z "$DEST" ]; then
  if [ "$IN_VENV" = 1 ] && ! use_system_python; then DEST="$POLYMATH_VENV/colibri"; else DEST="$POLYMATH_HOME/engines/colibri"; fi
fi
DEST="$(expand_home "$DEST")"
mkdir_user "$(dirname "$DEST")" "$STATE_DIR"
echo "  location: $DEST"

# ---------------------------------------------------------------- fetch upstream at the pinned commit
if [ ! -d "$DEST/.git" ]; then
  [ -e "$DEST" ] && [ -n "$(ls -A "$DEST" 2>/dev/null)" ] && die "$DEST exists and is not a git checkout: choose another --dir"
  say "== fetching colibri from $UPSTREAM"
  as_user git clone -q "$UPSTREAM" "$DEST"
fi
if ! as_user git -C "$DEST" cat-file -e "$PIN^{commit}" 2>/dev/null; then
  as_user git -C "$DEST" fetch -q origin || die "could not fetch $UPSTREAM"
fi
# undo an earlier patch (only the files the patches touch), then pin
for f in olmoe.c olmoe_cuda_chain.h qwen36.c qwen36_cuda_chain.h; do
  as_user git -C "$DEST" checkout -q "$PIN" -- "c/$f" 2>/dev/null || true
done
as_user git -C "$DEST" -c advice.detachedHead=false checkout -q "$PIN" || die "could not check out $PIN in $DEST (local changes? see git -C $DEST status)"
ok "colibri at $PIN"

# ---------------------------------------------------------------- patch
say "== applying the knowledge-pack patches"
for p in "$ENG"/patches/*.patch; do
  as_user git -C "$DEST" apply --whitespace=nowarn "$p" || die "patch $(basename "$p") does not apply to $PIN"
  ok "$(basename "$p")"
done
for f in "$ENG"/files/*; do as_user cp "$f" "$DEST/c/"; done
mkdir_user "$DEST/c/tests"
as_user cp "$ENG/tests/test_olmoe_overlay_swap.c" "$DEST/c/tests/"
ok "added $(ls "$ENG/files" | tr '\n' ' ')"
if [ "$WITH_PMC" = 1 ]; then
  say "== applying the polymath-c patches (Polymath packs in C, beside the B-slot hooks)"
  for p in "$PMC"/patches/*.patch; do
    as_user git -C "$DEST" apply --whitespace=nowarn "$p" || die "patch $(basename "$p") does not apply to $PIN (--no-polymath-c skips it)"
    ok "$(basename "$p")"
  done
  for f in "$PMC"/files/*; do as_user cp "$f" "$DEST/c/"; done
  ok "added $(ls "$PMC/files" | tr '\n' ' ')"
fi

# ---------------------------------------------------------------- build
# Each build goes to c/polymath-bin/<cpu|gpu>/; c/olmoe and c/qwen36 are left as the last one (GPU when built).
JOBS="$(nproc 2>/dev/null || echo 4)"
# A GPU build that fails does not stop the install: the CPU build still serves, and the errors go to one file to send
# back (logs/polymath-c-gpu-build.txt; ./scripts/polymath_gpu_report.sh adds the machine's details to it).
GPU_ERR="$POLYMATH_HOME/logs/polymath-c-gpu-build.txt"
mkdir_user "$POLYMATH_HOME/logs"
rm -f "$GPU_ERR"
BUILT=()
for B in "${BUILDS[@]}"; do
  if [ "$B" = gpu ]; then FLAGS=(CUDA=1 "CUDA_ARCH=$ARCH"); else FLAGS=(CUDA=0); fi
  say "== building olmoe and qwen36 ($B)"
  bok=1
  for T in olmoe qwen36; do
    MK=(make); [ "$REBUILD" = 1 ] && MK=(make -B)       # -B: every object compiled again, not only what changed
    if ( cd "$DEST/c" && as_user "${MK[@]}" -j"$JOBS" "$T" "${FLAGS[@]}" >"$DEST/build-$T-$B.log" 2>&1 ); then
      mkdir_user "$DEST/c/polymath-bin/$B"
      as_user cp -f "$DEST/c/$T" "$DEST/c/polymath-bin/$B/$T"
      ok "$T ($B)"
    else
      tail -20 "$DEST/build-$T-$B.log"
      [ "$B" = gpu ] || die "$T $B build failed (log: $DEST/build-$T-$B.log)"
      bok=0
      { echo "== $T GPU build failed $(date -u +%Y-%m-%dT%H:%M:%SZ)  (make $T ${FLAGS[*]})"
        echo "-- nvcc: $("$NVCC" --version 2>&1 | tail -2 | tr '\n' ' ')"; echo "-- gcc: $(gcc --version | head -1)"
        echo "-- the last 200 lines of $DEST/build-$T-$B.log:"; tail -200 "$DEST/build-$T-$B.log"; echo; } >> "$GPU_ERR"
      warn "$T GPU build failed: the CPU build still serves. The errors are in $GPU_ERR: run ./scripts/polymath_gpu_report.sh and send the file it writes"
      rm -f "$DEST/c/polymath-bin/gpu/$T"
    fi
  done
  [ "$bok" = 1 ] && BUILT+=("$B")
done
[ ${#BUILT[@]} -gt 0 ] || die "no build succeeded"
if [ "$WITH_PMC" = 1 ] && [ -x "$DEST/c/polymath-bin/gpu/qwen36" ]; then
  if grep -q "pm-chain-v1" "$DEST/c/polymath-bin/gpu/qwen36"; then ok "polymath-c(gpu): the CUDA chain port is in the GPU build (Settings → Engine → polymath-c-gpu)"
  else warn "the GPU build has no polymath-c(gpu) chain port"; fi
fi
if [ "$WITH_PMC" = 1 ]; then   # the chain port against the CPU path on colibri's fake CUDA backend: no GPU needed, seconds
  as_user cp "$PMC/tests/c/test_polymath_c_chain.c" "$DEST/c/tests/"
  if ( cd "$DEST/c" && as_user gcc -O2 -fopenmp -pthread -w -DCOLI_CUDA tests/test_polymath_c_chain.c -o tests/test_polymath_c_chain -lm \
       >"$DEST/build-chain-test.log" 2>&1 && OMP_NUM_THREADS=2 ./tests/test_polymath_c_chain >>"$DEST/build-chain-test.log" 2>&1 ); then
    ok "polymath-c(gpu) chain test (fake CUDA backend): $(tail -1 "$DEST/build-chain-test.log")"
  else
    warn "polymath-c(gpu) chain test failed: $DEST/build-chain-test.log (added to $GPU_ERR)"
    { echo "== the polymath-c(gpu) chain test failed $(date -u +%Y-%m-%dT%H:%M:%SZ)"; tail -80 "$DEST/build-chain-test.log"; echo; } >> "$GPU_ERR"
  fi
fi
[ -f "$GPU_ERR" ] && give_back "$GPU_ERR"
BUILDS=("${BUILT[@]}")
if [ "$RUN_TESTS" = 1 ]; then
  ( cd "$DEST/c" && as_user gcc -O2 -fopenmp -pthread -w tests/test_olmoe_overlay_swap.c -o tests/test_olmoe_overlay_swap -lm ) \
    && ok "overlay swap test compiled (run by packsmith/research/colibri/test_colibri.sh)" || warn "the swap test did not compile"
  grep -q coli_kn_apply "$DEST/c/qwen36.c" && grep -q olm_kn_apply "$DEST/c/olmoe.c" && ok "pack hooks present" || die "pack hooks missing after patching"
  if [ "$WITH_PMC" = 1 ]; then
    grep -q pm_merge_row "$DEST/c/qwen36.c" && grep -q pm_merge_row "$DEST/c/olmoe.c" && ok "polymath-c hooks present" || die "polymath-c hooks missing after patching"
    echo "  polymath-c tests (CPU, tiny models, a few minutes; need the Polymath Engine for the reference side):"
    echo "    $(venv_py) $PMC/tests/test_polymath_c.py --colibri $DEST      parity with PyTorch, exactness, refusals"
    echo "    $(venv_py) $PMC/tests/test_live_swap.py --colibri $DEST       live swaps in one serving process"
  fi
fi

# ---------------------------------------------------------------- record
PATCH_VERSION="$(engine_fingerprint "$HERE" "$WITH_PMC" "$REQ_MODE")"
GPU_FAILED=false; [[ " ${BUILDS[*]} " == *" gpu "* ]] && [[ " ${BUILT[*]} " != *" gpu "* ]] && GPU_FAILED=true
cat > "$DEST/c/polymath_engine.json" <<JSON
{"engine": "patched-colibri", "polymath_c": $([ "$WITH_PMC" = 1 ] && echo true || echo false), "upstream": "$UPSTREAM", "commit": "$PIN", "patch_version": "$PATCH_VERSION",
 "mode": "$MODE", "requested": "$REQ_MODE", "builds": [$(printf '"%s",' "${BUILDS[@]}" | sed 's/,$//')], "built_ok": [$(printf '"%s",' "${BUILT[@]}" | sed 's/,$//')],
 "gpu_failed": $GPU_FAILED, "rebuild": $([ "$REBUILD" = 1 ] && echo true || echo false),
 "cuda_arch": "$([ "$MODE" != cpu ] && echo "$ARCH")", "built": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
JSON
cp "$DEST/c/polymath_engine.json" "$STATE_DIR/colibri.json"
echo "$DEST" > "$STATE_DIR/colibri_dir"
give_back "$DEST/c/polymath_engine.json" "$STATE_DIR"
# tell Packsmith where it is (when Packsmith is already installed)
if [ -x "$(venv_py)" ] && as_user "$(venv_py)" -c "import packsmith" 2>/dev/null; then
  as_user "$(venv_py)" -m packsmith.system.setup set "colibri_dir=$DEST" && ok "Packsmith → colibri_dir = $DEST"
fi
say "Done: patched colibri (${BUILDS[*]})$([ "$WITH_PMC" = 1 ] && echo " with polymath-c") in $DEST"
echo "  undo the patch:  git -C $DEST checkout $PIN -- c && git -C $DEST clean -fd c"
