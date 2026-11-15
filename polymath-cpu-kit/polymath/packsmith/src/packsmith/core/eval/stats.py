"""Bootstrap confidence intervals and seed aggregation."""
from __future__ import annotations

import random


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(xs: list[float], n: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    if not xs:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    ms = sorted(mean([xs[rng.randrange(len(xs))] for _ in xs]) for _ in range(n))
    return mean(xs), ms[int(n * alpha / 2)], ms[int(n * (1 - alpha / 2)) - 1]


def overlap(a: tuple, b: tuple) -> bool:
    return not (a[1] > b[2] or b[1] > a[2])
