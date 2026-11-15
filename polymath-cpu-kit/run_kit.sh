#!/usr/bin/env bash
# Polymath CPU test kit: does a knowledge pack teach an MoE model something, as served by patched colibri?
# CPU only, no GPU or CUDA needed. Fetches colibri at the pinned commit, applies Polymath's pack patches, builds the
# CPU engine, trains one pack on a small model and prints base versus base + pack.
#
#   bash run_kit.sh                    the small OLMoE-layout model shipped in model/ (about 20 to 40 minutes)
#   bash run_kit.sh olmoe              the real OLMoE-1B-7B-0125 from Hugging Face (14 GB, 32 GB RAM or more, many hours)
#   bash run_kit.sh /path/to/folder    any OLMoE model folder in Hugging Face format
#   options:  --tests N    questions per row (default: every question on the small model, 40 on a real one)    --threads N   CPU threads (default: all)
#             --retrain    train the pack again              --yes         no questions
#
# Everything goes to ~/.polymath-kit (POLYMATH_KIT_HOME to change it): the venv, colibri, the model download and the
# runs. Your own Polymath install (~/.polymath) is not touched. Under sudo it is still your home, owned by you.
# The result is printed at the end and written to ~/.polymath-kit/runs/<model>/RESULTS.md.
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
POLY="$KIT/polymath"
# shellcheck source=polymath/scripts/lib/common.sh
source "$POLY/scripts/lib/common.sh"

MODEL=small; TESTS=""; THREADS="$(nproc 2>/dev/null || echo 4)"; RETRAIN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --tests) TESTS="$2"; shift ;;
    --threads) THREADS="$2"; shift ;;
    --retrain) RETRAIN=1 ;;
    --yes|-y) ASSUME_YES=1 ;;
    -h|--help) sed -n 2,15p "$0"; exit 0 ;;
    qwen|qwen3) die "Qwen3-30B-A3B is not in the CPU kit (about 60 GB of weights to train on a CPU). On a GPU use polymath's scripts/regression/run_qwen.sh" ;;
    -*) die "unknown option $1 (see --help)" ;;
    *) MODEL="$1" ;;
  esac
  shift
done

KIT_HOME="$(expand_home "${POLYMATH_KIT_HOME:-$REAL_HOME/.polymath-kit}")"
export POLYMATH_HOME="$KIT_HOME"                      # colibri's installer writes its state here, not in ~/.polymath
export POLYMATH_VENV="$KIT_HOME/venv"
COLI="$KIT_HOME/colibri"
mkdir_user "$KIT_HOME" "$KIT_HOME/runs"
CORES="$THREADS"

# ---------------------------------------------------------------- what will run, and how long
case "$MODEL" in
  small) SRC="$KIT/model/small-olmoe"; LABEL="small OLMoE layout (6 layers, 16 experts top-4, 21.5M weights, kit model)"
         EST="about 10 to 30 minutes on $CORES threads, plus about 5 the first time (PyTorch download, colibri build)" ;;
  olmoe) SRC=""; LABEL="allenai/OLMoE-1B-7B-0125"
         EST="many hours on a CPU (the 7B pack training alone is hours, each answer seconds); 14 GB download the first time" ;;
  *)     SRC="$(cd "$(expand_home "$MODEL")" 2>/dev/null && pwd)" || die "$MODEL: not small, olmoe or a folder"
         [ -f "$SRC/config.json" ] || die "$SRC has no config.json (a Hugging Face model folder is needed)"
         grep -q '"model_type": *"olmoe"' "$SRC/config.json" || die "$SRC is not an OLMoE model (this kit serves OLMoE only)"
         LABEL="$(basename "$SRC")"; EST="depends on the model's size" ;;
esac
say "== Polymath CPU test kit"
echo "  model:    $LABEL"
echo "  expected: $EST"
echo "  steps:    Python venv, patched colibri (CPU build), dataset, pack training, colibri answers base and base + pack"
if [ "$MODEL" = olmoe ]; then
  MEM_GB=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1048576 ))
  [ "$MEM_GB" -ge 30 ] || warn "this machine has ${MEM_GB} GB of RAM; the 7B model needs about 32 GB to train on the CPU"
  case "$(ask "continue with the real OLMoE on the CPU? (y/n)" y)" in y*|Y*) ;; *) exit 0 ;; esac
fi
T_START=$(date +%s)

# ---------------------------------------------------------------- 1. tools and the venv
say "== 1. Python venv ($POLYMATH_VENV)"
for t in git gcc make; do command -v "$t" >/dev/null || die "missing: $t   (sudo apt install git build-essential)"; done
PYBIN="$(find_python)"
if [ -z "$PYBIN" ]; then          # any Python 3.12+ will do for the kit
  for c in python3.14 python3.13 python3.12 python3; do
    command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 12))' && { PYBIN="$(command -v "$c")"; break; }
  done
fi
[ -n "$PYBIN" ] || die "Python 3.12 or newer is needed (Ubuntu 26.04: sudo apt install python3 python3-venv)"
PY="$POLYMATH_VENV/bin/python"
if [ ! -x "$PY" ]; then
  as_user "$PYBIN" -m venv "$POLYMATH_VENV" || die "could not create the venv (sudo apt install python3-venv)"
fi
if ! "$PY" -c "import torch, transformers, safetensors, tokenizers" 2>/dev/null; then
  as_user "$PY" -m pip install -q --upgrade pip
  echo "  PyTorch for the CPU (about 200 MB)..."
  as_user "$PY" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu \
    || { warn "the PyTorch CPU index was not reachable: installing torch from PyPI (larger)"; as_user "$PY" -m pip install -q torch; }
  as_user "$PY" -m pip install -q "transformers>=5.0" "safetensors>=0.4" "tokenizers>=0.21" "numpy>=2.0" "huggingface_hub>=0.30" pyyaml
fi
ok "$("$PY" -c 'import sys, torch, transformers; print("Python", sys.version.split()[0], "| torch", torch.__version__, "| transformers", transformers.__version__)')"

# ---------------------------------------------------------------- 2. patched colibri, CPU build
say "== 2. patched colibri (CPU build) in $COLI"
WANT="$(engine_fingerprint "$POLY" 0 cpu)"
HAVE="$(sed -n 's/.*"patch_version": *"\([^"]*\)".*/\1/p' "$COLI/c/polymath_engine.json" 2>/dev/null || true)"
if [ -x "$COLI/c/polymath-bin/cpu/olmoe" ] && [ "$HAVE" = "$WANT" ]; then
  ok "already built with these patches ($WANT)"
else
  bash "$POLY/install_patched_colibri.sh" --cpu --yes --no-polymath-c --dir "$COLI" > "$KIT_HOME/colibri_build.log" 2>&1 \
    || { tail -25 "$KIT_HOME/colibri_build.log"; die "the colibri build failed: $KIT_HOME/colibri_build.log"; }
  grep -E "✓|!" "$KIT_HOME/colibri_build.log" | sed 's/^/ /' | head -12
fi

# ---------------------------------------------------------------- 3. the model
say "== 3. model"
if [ "$MODEL" = olmoe ]; then
  export HF_HUB_CACHE="$POLYMATH_HF_CACHE"
  SRC="$(as_user "$PY" -c 'import sys; from huggingface_hub import snapshot_download as s; print(s("allenai/OLMoE-1B-7B-0125"))' | tail -1)" \
    || die "the download failed (network, or log in with huggingface-cli login)"
fi
ok "$LABEL"
RUN="$KIT_HOME/runs/$( [ "$MODEL" = small ] && echo small-olmoe || basename "$SRC" | tr -c 'A-Za-z0-9._-\n' _)"
mkdir_user "$RUN"

# ---------------------------------------------------------------- 4. train, convert, ask, report
say "== 4. pack and results (run folder $RUN)"
args=(--model "$SRC" --model-label "$LABEL" --colibri "$COLI" --out "$RUN" --threads "$THREADS")
[ -n "$TESTS" ] && args+=(--tests "$TESTS")
[ "$RETRAIN" = 1 ] && args+=(--retrain)
as_user env OMP_NUM_THREADS="$THREADS" "$PY" "$KIT/kit/kit.py" "${args[@]}" 2>&1 | tee "$RUN/kit.log"
rc=${PIPESTATUS[0]}
give_back "$KIT_HOME"
echo "  total $(( ($(date +%s) - T_START) / 60 )) min"
[ "$rc" = 0 ] && echo "  results: $RUN/RESULTS.md   (results.json has every question and reply)" \
              || echo "  something failed: send $RUN/kit.log and the logs folder beside it"
exit "$rc"
