from __future__ import annotations

from typing import Dict, List, Tuple
import math
import numpy as np


TYPE_ORDER = ("PE", "EC", "SI")


def _normalize_ratios(ratios: Tuple[float, float, float]) -> Tuple[float, float, float]:
    a, b, c = [max(0.0, float(x)) for x in ratios]
    s = a + b + c
    if s <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return (a / s, b / s, c / s)


def _make_quotas(k_keep: int, ratios: Tuple[float, float, float]) -> List[int]:
    ratios = _normalize_ratios(ratios)
    quotas = [int(math.floor(r * k_keep)) for r in ratios]

    while sum(quotas) < k_keep:
        i = int(np.argmin(quotas))
        quotas[i] += 1
    while sum(quotas) > k_keep:
        i = int(np.argmax(quotas))
        quotas[i] -= 1
    return quotas


def _assign_relevance(cands: List[Dict]) -> None:

    for src in ("image", "question", "caption"):
        xs = [float(c.get("score", 0.0)) for c in cands if c.get("src") == src]
        if not xs:
            continue
        mu = float(np.mean(xs))
        sd = float(np.std(xs) + 1e-6)
        hi = mu + 0.5 * sd
        lo = mu - 0.5 * sd

        for c in cands:
            if c.get("src") != src:
                continue
            s = float(c.get("score", 0.0))
            if s >= hi:
                c["relevance"] = "High"
            elif s <= lo:
                c["relevance"] = "Low"
            else:
                c["relevance"] = "Medium"


def postprocess_triplets(
    candidates: List[Dict],
    *,
    k_keep: int = 6,
    tau: float = 0.2,
    ratios: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),  # (PE, EC, SI)
) -> List[Dict]:

    k_keep = int(k_keep)
    tau = float(tau)


    filtered = [c for c in candidates if float(c.get("score", 0.0)) >= tau]


    best: Dict[int, Dict] = {}
    for c in filtered:
        idx = int(c.get("id", -1))
        if idx < 0:
            continue
        if idx not in best or float(c.get("score", 0.0)) > float(best[idx].get("score", 0.0)):
            best[idx] = c
    cands = list(best.values())
    if not cands:
        return []

    _assign_relevance(cands)


    quotas = _make_quotas(k_keep, ratios)
    buckets = {t: [] for t in TYPE_ORDER}
    for c in sorted(cands, key=lambda x: float(x.get("score", 0.0)), reverse=True):
        t = str(c.get("type", "")).upper()
        if t in buckets:
            buckets[t].append(c)

    kept: List[Dict] = []
    for t, q in zip(TYPE_ORDER, quotas):
        kept.extend(buckets[t][:q])

    kept = sorted(kept, key=lambda x: float(x.get("score", 0.0)), reverse=True)[:k_keep]
    return kept
