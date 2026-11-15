"""Store a model folder's weights in bfloat16 (half the size; what the kit ships). python kit/to_bf16.py <folder>"""
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

p = Path(sys.argv[1]) / "model.safetensors"
t = load_file(str(p))
save_file({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in t.items()}, str(p), metadata={"format": "pt"})
cfg = Path(sys.argv[1]) / "config.json"
import json
c = json.loads(cfg.read_text())
c["dtype"] = "bfloat16"
c.pop("torch_dtype", None)
cfg.write_text(json.dumps(c, indent=2))
print(f"{p}: bfloat16, {p.stat().st_size / 1e6:.0f} MB")
