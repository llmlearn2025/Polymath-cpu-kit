#!/usr/bin/env bash
# Rebuild the kit's small model from nothing (CPU, no download): the dataset, a 2,000-piece vocabulary and an empty
# OLMoE-layout model (6 layers, 16 experts top-4, hidden 256, 21.5M weights), pretrained on the invented "base world"
# only. The pack's harbour world is never in it. Run run_kit.sh once first (it makes the venv).
#
#   bash build_small_model.sh [minutes]      default 100; writes ~/.polymath-kit/models/small-olmoe-rebuilt
#   then:  bash run_kit.sh ~/.polymath-kit/models/small-olmoe-rebuilt
#
# Pretraining runs for a fixed time, so a rebuild on another machine sees a different number of tokens and is not
# bit-identical to the shipped model in model/small-olmoe (its pretrain.json records steps, tokens and loss).
set -euo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
POLY="$KIT/polymath"
source "$POLY/scripts/lib/common.sh"
MIN="${1:-100}"
KIT_HOME="$(expand_home "${POLYMATH_KIT_HOME:-$REAL_HOME/.polymath-kit}")"
PY="$KIT_HOME/venv/bin/python"
[ -x "$PY" ] || die "no venv yet: run bash run_kit.sh first"
OUT="$KIT_HOME/models/small-olmoe-rebuilt"; DATA="$KIT_HOME/models/data"
mkdir_user "$KIT_HOME/models"
as_user env PYTHONPATH="$POLY/packsmith/src" "$PY" "$POLY/scripts/regression/make_data.py" --out "$DATA"
say "== pretraining for $MIN minutes on $(nproc) threads"
as_user env PYTHONPATH="$POLY/packsmith/src" "$PY" "$POLY/scripts/regression/pretrain_tiny.py" --data "$DATA" --out "$OUT" \
  --minutes "$MIN" --threads "$(nproc)"
as_user env PYTHONPATH="$POLY/packsmith/src" "$PY" "$KIT/kit/to_bf16.py" "$OUT"
give_back "$KIT_HOME/models"
ok "model: $OUT   (bash run_kit.sh $OUT)"
