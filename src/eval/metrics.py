from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


def _log2(x: float) -> float:
    return math.log(x, 2)


def dcg_at_k(rels: list[int], k: int) -> float:
    s = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        if rel <= 0:
            continue
        s += (2.0**rel - 1.0) / _log2(i + 1.0)
    return s


def ndcg_at_k(rels: list[int], k: int) -> float:
    dcg = dcg_at_k(rels, k)
    ideal = sorted(rels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return 0.0 if idcg == 0 else dcg / idcg


def mrr_at_k(ranked_docids: list[str], relevant: set[str], k: int) -> float:
    for i, d in enumerate(ranked_docids[:k], start=1):
        if d in relevant:
            return 1.0 / float(i)
    return 0.0


def recall_at_k(ranked_docids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hit = any(d in relevant for d in ranked_docids[:k])
    return 1.0 if hit else 0.0


@dataclass(frozen=True)
class IRMetrics:
    mrr_10: float
    ndcg_10: float
    recall_50: float
    recall_100: float
    recall_1000: float


def compute_metrics(
    rankings: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
) -> IRMetrics:
    mrrs = []
    ndcgs = []
    r50 = []
    r100 = []
    r1000 = []

    for qid, ranked in rankings.items():
        relmap = qrels.get(qid, {})
        relevant = {d for d, r in relmap.items() if r > 0}
        if not relevant:
            continue  # skip unjudged queries (standard IR evaluation practice)
        mrrs.append(mrr_at_k(ranked, relevant, 10))

        rels = [int(relmap.get(d, 0)) for d in ranked[:10]]
        ndcgs.append(ndcg_at_k(rels, 10))

        r50.append(recall_at_k(ranked, relevant, 50))
        r100.append(recall_at_k(ranked, relevant, 100))
        r1000.append(recall_at_k(ranked, relevant, 1000))

    n = max(1, len(mrrs))
    return IRMetrics(
        mrr_10=float(sum(mrrs) / n),
        ndcg_10=float(sum(ndcgs) / n),
        recall_50=float(sum(r50) / n),
        recall_100=float(sum(r100) / n),
        recall_1000=float(sum(r1000) / n),
    )

