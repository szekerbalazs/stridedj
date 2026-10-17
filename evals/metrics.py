"""Aggregate metrics over trials.

pass@k: probability that at least one of k trials succeeds ("can it do it at all?"). pass^k: probability that all k
trials succeed ("can I rely on it every time?"). For a user-facing agent, pass^k is the one that matters: with a
75 % per-trial success rate, pass^3 = 0.75³ ≈ 42 %.

With n trials of which c passed, the unbiased estimators are (Chen et al. 2021 for pass@k):
    pass@k = 1 − C(n−c, k) / C(n, k)        pass^k = C(c, k) / C(n, k)
"""

from __future__ import annotations

from collections.abc import Sequence
from math import comb

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("need 0 <= c <= n and 1 <= k <= n")
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("need 0 <= c <= n and 1 <= k <= n")
    return comb(c, k) / comb(n, k)


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else float("nan")
