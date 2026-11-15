# Shared helpers for Polymath's scripts (sourced, not run).
#
# Under sudo every path is the REAL user's (the one who typed sudo), never /root, and whatever the script creates is
# handed back to that user. Python is the venv's (or the system's with POLYMATH_VENV=system),
# always called by its full path: never a bare `python`.

POLY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

say()  { printf '\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- the real user (also under sudo)
if [ "$(id -u)" = 0 ] && [ -n "${SUDO_USER:-}" ] && [ "${SUDO_USER}" != root ]; then
  REAL_USER="$SUDO_USER"
  REAL_HOME="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
  REAL_GROUP="$(id -gn "$SUDO_USER")"
  UNDER_SUDO=1
else
  REAL_USER="$(id -un)"
  REAL_HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
  REAL_GROUP="$(id -gn)"
  UNDER_SUDO=0
fi
export REAL_USER REAL_HOME

# give_back PATH...  — under sudo, chown what we created to the real user
give_back() {
  [ "$UNDER_SUDO" = 1 ] || return 0
  for p in "$@"; do [ -e "$p" ] && chown -R "$REAL_USER:$REAL_GROUP" "$p" 2>/dev/null || true; done
}
# as_user CMD...  — run a command as the real user (a no-op wrapper when not under sudo)
as_user() {
  if [ "$UNDER_SUDO" = 1 ]; then
    sudo -u "$REAL_USER" -H --preserve-env=POLYMATH_HOME,POLYMATH_VENV,POLYMATH_HF_CACHE,POLYMATH_PACKS,POLYMATH_MODELS,HF_TOKEN,HF_HOME,HF_HUB_CACHE,PATH "$@"
  else
    "$@"
  fi
}
# mkdir_user DIR...  — mkdir -p, owned by the real user
mkdir_user() {
  for d in "$@"; do
    if [ ! -d "$d" ]; then
      if [ "$UNDER_SUDO" = 1 ]; then sudo -u "$REAL_USER" mkdir -p "$d" 2>/dev/null || { mkdir -p "$d"; give_back "$d"; }
      else mkdir -p "$d"; fi
    fi
  done
}
expand_home() { local p="$1"; case "$p" in "~") p="$REAL_HOME" ;; "~/"*) p="$REAL_HOME/${p#~/}" ;; esac; echo "$p"; }

# ---------------------------------------------------------------- locations
POLYMATH_HOME="$(expand_home "${POLYMATH_HOME:-$REAL_HOME/.polymath}")"
POLYMATH_VENV="${POLYMATH_VENV:-$POLY_ROOT/.venv}"; [ "$POLYMATH_VENV" = system ] || POLYMATH_VENV="$(expand_home "$POLYMATH_VENV")"
POLYMATH_HF_CACHE="$(expand_home "${POLYMATH_HF_CACHE:-${HF_HUB_CACHE:-$REAL_HOME/.cache/huggingface/hub}}")"
POLYMATH_PACKS="$(expand_home "${POLYMATH_PACKS:-$POLYMATH_HOME/packs}")"
POLYMATH_MODELS="$(expand_home "${POLYMATH_MODELS:-$POLYMATH_HOME/models}")"
export POLYMATH_HOME POLYMATH_VENV POLYMATH_HF_CACHE POLYMATH_PACKS POLYMATH_MODELS
STATE_DIR="$POLYMATH_HOME/state"          # marker files: what is installed, from which version
PY_WANT="${POLYMATH_PYTHON_VERSION:-3.14}"

# POLYMATH_VENV=system: no venv, the system Python (Docker uses this; also fine on a machine of its own)
use_system_python() { [ "$POLYMATH_VENV" = system ]; }
# find_python: a Python $PY_WANT on PATH. Ubuntu 26.04 has it as python3 (and python3.14); plain `python` may not exist.
find_python() {
  local c v
  for c in "python$PY_WANT" python3 python; do
    command -v "$c" >/dev/null 2>&1 || continue
    v="$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
    [ "$v" = "$PY_WANT" ] && { command -v "$c"; return; }
  done
  echo ""
}
venv_py() { if use_system_python; then find_python; else echo "$POLYMATH_VENV/bin/python"; fi; }

# ---------------------------------------------------------------- GPU / CUDA (installing them is the user's job)
has_nvidia_driver() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }
gpu_list() { nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null; }
gpu_count() { has_nvidia_driver && gpu_list | grep -c . || echo 0; }
find_nvcc() {
  if command -v nvcc >/dev/null 2>&1; then command -v nvcc; return; fi
  for d in "${CUDA_HOME:-}" /usr/local/cuda /usr/local/cuda-*; do [ -x "$d/bin/nvcc" ] && { echo "$d/bin/nvcc"; return; }; done
  echo ""
}
cuda_driver_version() { nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9]*\).*/\1/p' | head -1; }
cuda_help() {
  cat >&2 <<'EOF'
  The NVIDIA driver and CUDA toolkit are not installed by Polymath. On Ubuntu:
    driver:  sudo ubuntu-drivers install            (then reboot; check with nvidia-smi)
    toolkit: https://developer.nvidia.com/cuda-downloads  (nvcc; needed only to BUILD colibri for the GPU)
  Or run the CPU build:  --cpu   (works everywhere, much slower), or use Docker (the GPU image has the toolkit).
EOF
}

# ---------------------------------------------------------------- ask (skipped with --yes / no terminal)
ASSUME_YES="${ASSUME_YES:-0}"
ask() {   # ask "question" default -> echoes the answer
  local q="$1" def="$2" a=""
  if [ "$ASSUME_YES" = 1 ] || [ ! -t 0 ]; then echo "$def"; return; fi
  read -r -p "  $q [$def] " a </dev/tty || true
  echo "${a:-$def}"
}

# The patched-colibri build's fingerprint: what decides whether the engines must be built again. The patches and added
# files (B-slot, and polymath-c unless left out), the installer itself (build steps and flags), and the build asked
# for (cpu / gpu / both / auto). run.sh compares it with the one the last install recorded.
#   engine_fingerprint <repo> <with polymath-c: 1|0> <requested mode>
engine_fingerprint() {
  local root="$1" pmc="$2" mode="$3" extra=()
  [ "$pmc" = 1 ] && [ -d "$root/engines/polymath-c/patches" ] && extra=("$root"/engines/polymath-c/patches/*.patch "$root"/engines/polymath-c/files/*)
  { cat "$root"/engines/patched-colibri/patches/*.patch "$root"/engines/patched-colibri/files/* "${extra[@]}" \
        "$root/install_patched_colibri.sh" "$root/engines/patched-colibri/UPSTREAM_COMMIT"; echo "mode=$mode pmc=$pmc"; } \
    | sha256sum | cut -c1-12
}
