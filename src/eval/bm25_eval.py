from __future__ import annotations

import logging
import re
from dataclasses import asdict
from typing import Iterable

from rank_bm25 import BM25Okapi

from src.eval.metrics import IRMetrics, compute_metrics

log = logging.getLogger(__name__)


_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _tokenize(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


def bm25_rank(
    *,
    passages: list[tuple[str, str]],
    queries: list[tuple[str, str]],
    qrels: dict[str, dict[str, int]],
    topk: int,
) -> tuple[dict[str, list[str]], IRMetrics]:
    """
    BM25 baseline using rank-bm25.

    This is intended for toy/smoke and small corpora. For MS MARCO scale, use a Lucene/Pyserini index.
    """
    if len(passages) > 100_000:
        raise RuntimeError(
            f"Corpus too large for pure-Python BM25 (N={len(passages)}). "
            "Use Pyserini/Lucene for MS MARCO scale."
        )

    docids = [d for d, _ in passages]
    corpus = [_tokenize(t) for _, t in passages]
    bm25 = BM25Okapi(corpus)

    rankings: dict[str, list[str]] = {}
    for qid, qtext in queries:
        scores = bm25.get_scores(_tokenize(qtext))
        # argsort descending
        import numpy as np

        idx = np.argpartition(-scores, kth=min(topk, len(scores) - 1))[:topk]
        idx = idx[np.argsort(-scores[idx])]
        rankings[qid] = [docids[i] for i in idx.tolist()]

    metrics = compute_metrics(rankings, qrels)
    return rankings, metrics

