"""Minimal BM25 (no dependencies) for the RAG and DMoE-style baselines."""
from __future__ import annotations

import math
import re
from collections import Counter


def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.tf = [Counter(_tok(d)) for d in docs]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / max(len(docs), 1)
        df = Counter(w for t in self.tf for w in t)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.k1, self.b = k1, b

    def scores(self, q: str) -> list[float]:
        qs = _tok(q)
        out = []
        for tf, L in zip(self.tf, self.len):
            s = 0.0
            for w in qs:
                f = tf.get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.avg))
            out.append(s)
        return out

    def top(self, q: str, k: int) -> list[int]:
        sc = self.scores(q)
        return sorted(range(len(sc)), key=lambda i: -sc[i])[:k]
