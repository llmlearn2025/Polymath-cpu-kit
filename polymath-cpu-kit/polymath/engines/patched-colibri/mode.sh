# Sourced by the colibri scripts: choose where colibri computes.
#   cpu=1 (default)  experts and everything else on the CPU
#   gpu=1            CUDA build: the whole model on the RTX card (colibri's CUDA chain)
# Upper case (GPU=1 / CPU=1) works too.
GPU="${GPU:-${gpu:-0}}"
if [ "${CPU:-${cpu:-0}}" = 1 ]; then GPU=0; fi
if [ "$GPU" = 1 ]; then
  export COLI_CUDA=1
  # colibri keeps attention/head weights as half-precision "twins" on the CPU by default
  # (COLI_DENSE_F16=1); the CUDA chain cannot upload those and falls back to the CPU.
  export COLI_DENSE_F16=0
  COLI_MODE=gpu
  COLI_MAKE="make olmoe CUDA=1"
else
  export COLI_CUDA=0
  COLI_MODE=cpu
  COLI_MAKE="make olmoe"
fi
echo "== colibri mode: $COLI_MODE"
