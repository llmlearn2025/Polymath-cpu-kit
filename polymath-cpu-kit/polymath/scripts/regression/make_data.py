"""The regression dataset (the Train Lab's yardstick set): the generator is packsmith/lab/yardstick.py.

    python scripts/regression/make_data.py --out <dir> [--seed 0]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packsmith" / "src"))
from packsmith.lab.yardstick import main  # noqa: E402

if __name__ == "__main__":
    main()
