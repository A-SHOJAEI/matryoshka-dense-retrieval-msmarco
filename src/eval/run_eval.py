from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer

from src.data.io import read_jsonl, read_qrels_tsv, write_json
from src.data.stores import open_passage_store
from src.eval.bm25_eval import bm25_rank
from src.eval.metrics import compute_metrics
from src.model.encoder import load_biencoder
from src.utils.config import deep_get, load_yaml
from src.utils.logging import setup_logging

log = logging.getLogger(__name__)


def _processed_dir(cfg: dict) -> Path:
    dataset = deep_get(cfg, "data.dataset", required=True)
    if dataset == "toy":
        return Path(deep_get(cfg, "paths.data_dir", default="data")) / "processed" / "toy"
    if dataset == "msmarco":
        return Path(deep_get(cfg, "data.msmarco.processed_dir", required=True))
    raise ValueError(f"Unknown dataset: {dataset}")


def _load_queries(path: Path, *, max_queries: Optional[int]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for row in read_jsonl(path):
        out.append((str(row["qid"]), str(row["text"])))
        if max_queries is not None and len(out) >= max_queries:
            break
    return out


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels_rows = read_qrels_tsv(path)
    qrels: dict[str, dict[str, int]] = {}
    for qid, docid, rel in qrels_rows:
        qrels.setdefault(qid, {})[str(docid)] = int(rel)
    return qrels


def _load_passages(processed: Path, *, max_passages: Optional[int]) -> list[tuple[str, str]]:
    # For evaluation on toy/small corpora. For MS MARCO scale, use pre-encoded embeddings + ANN.
    passages_path = processed / "passages.jsonl"
    if not passages_path.exists():
        raise FileNotFoundError(f"Missing passages.jsonl at {passages_path}")
    out: list[tuple[str, str]] = []
    for row in read_jsonl(passages_path):
        out.append((str(row["docid"]), str(row["text"])))
        if max_passages is not None and len(out) >= max_passages:
            break
    return out


def _parse_top1000_dev(path: Path) -> dict[str, list[str]]:
    """
    Parse MS MARCO BM25 top1000 candidate file.

    Common formats are space- or tab-separated with at least (qid, docid, ...).
    """
    cand: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split()
            if len(cols) < 2:
                continue
            qid, docid = cols[0], cols[1]
            cand.setdefault(str(qid), []).append(str(docid))
    return cand


@torch.no_grad()
def _encode_texts(
    model,
    tokenizer,
    texts: list[str],
    *,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    embs: list[np.ndarray] = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        toks = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        e = model.encoder(**toks) if hasattr(model, "encoder") else model(**toks)  # defensive
        embs.append(e.detach().float().cpu().numpy())
    return np.concatenate(embs, axis=0)


def _truncate_renorm(x: np.ndarray, d: int) -> np.ndarray:
    y = x[:, :d]
    n = np.linalg.norm(y, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return y / n


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def evaluate_dense_exact(
    *,
    ckpt_dir: Path,
    passages: list[tuple[str, str]],
    queries: list[tuple[str, str]],
    qrels: dict[str, dict[str, int]],
    dims: list[int],
    device: torch.device,
    max_length_query: int,
    max_length_passage: int,
    batch_size: int = 32,
    topk: int = 1000,
) -> dict:
    model, meta = load_biencoder(str(ckpt_dir), map_location=device)
    model = model.to(device)
    tok = AutoTokenizer.from_pretrained(str(ckpt_dir / "tokenizer")) if (ckpt_dir / "tokenizer").exists() else AutoTokenizer.from_pretrained(meta["model"]["hf_model_name"])

    docids = [d for d, _ in passages]
    p_texts = [t for _, t in passages]
    qids = [q for q, _ in queries]
    q_texts = [t for _, t in queries]

    t0 = time.perf_counter()
    p_full = _encode_texts(model.encoder, tok, p_texts, max_length=max_length_passage, batch_size=batch_size, device=device)
    t1 = time.perf_counter()
    q_full = _encode_texts(model.encoder, tok, q_texts, max_length=max_length_query, batch_size=batch_size, device=device)
    t2 = time.perf_counter()

    per_dim: dict[str, dict] = {}
    for d in dims:
        p = _truncate_renorm(p_full, d)
        q = _truncate_renorm(q_full, d)

        rankings: dict[str, list[str]] = {}
        search_t = 0.0
        for qi, qid in enumerate(qids):
            s0 = time.perf_counter()
            scores = p @ q[qi]  # [N]
            idx = _topk_indices(scores, topk)
            rankings[qid] = [docids[i] for i in idx.tolist()]
            search_t += time.perf_counter() - s0

        m = compute_metrics(rankings, qrels)
        per_dim[str(d)] = {
            "metrics": asdict(m),
            "latency": {
                "encode_passages_s": float(t1 - t0),
                "encode_queries_s": float(t2 - t1),
                "search_total_s": float(search_t),
                "search_avg_ms_per_query": float((search_t / max(1, len(qids))) * 1000.0),
            },
            "memory": {
                "passage_matrix_bytes": int(p.shape[0] * p.shape[1] * 4),
            },
        }

    return {"checkpoint": str(ckpt_dir), "meta": meta, "per_dim": per_dim}


@torch.no_grad()
def evaluate_dense_candidates(
    *,
    ckpt_dir: Path,
    store,
    candidates: dict[str, list[str]],
    queries: list[tuple[str, str]],
    qrels: dict[str, dict[str, int]],
    dims: list[int],
    device: torch.device,
    max_length_query: int,
    max_length_passage: int,
    batch_size: int = 32,
    topk: int = 1000,
) -> dict:
    """
    Rerank candidate passages per query (e.g., MS MARCO BM25 top1000) instead of searching the full corpus.
    This avoids embedding the full 8.8M passage collection, but metrics are computed on the candidate set.
    """
    model, meta = load_biencoder(str(ckpt_dir), map_location=device)
    model = model.to(device)
    tok = (
        AutoTokenizer.from_pretrained(str(ckpt_dir / "tokenizer"))
        if (ckpt_dir / "tokenizer").exists()
        else AutoTokenizer.from_pretrained(meta["model"]["hf_model_name"])
    )

    qids = [q for q, _ in queries]
    q_texts = [t for _, t in queries]
    t0 = time.perf_counter()
    q_full = _encode_texts(model.encoder, tok, q_texts, max_length=max_length_query, batch_size=batch_size, device=device)
    t1 = time.perf_counter()

    # Pre-encode all candidate passages once (avoid re-encoding per dim).
    log.info("Encoding candidate passages for %d queries...", len(qids))
    query_passage_cache: dict[str, tuple[list[str], np.ndarray]] = {}
    for qi, qid in enumerate(qids):
        docids = candidates.get(qid, [])
        if not docids:
            continue
        texts: list[str] = []
        kept_docids: list[str] = []
        for docid_s in docids:
            p = store.get(int(docid_s))
            if p is None:
                continue
            kept_docids.append(str(p.docid))
            texts.append(p.text)
        if not texts:
            continue
        p_full = _encode_texts(
            model.encoder, tok, texts, max_length=max_length_passage, batch_size=batch_size, device=device
        )
        query_passage_cache[qid] = (kept_docids, p_full)
        if (qi + 1) % 100 == 0:
            log.info("  Encoded passages for %d/%d queries", qi + 1, len(qids))
    t2 = time.perf_counter()
    log.info("Passage encoding done in %.1fs for %d queries", t2 - t1, len(query_passage_cache))

    per_dim: dict[str, dict] = {}
    for d in dims:
        q = _truncate_renorm(q_full, d)
        rankings: dict[str, list[str]] = {}
        search_t = 0.0
        for qi, qid in enumerate(qids):
            if qid not in query_passage_cache:
                rankings[qid] = []
                continue
            kept_docids, p_full = query_passage_cache[qid]
            s0 = time.perf_counter()
            p = _truncate_renorm(p_full, d)
            scores = p @ q[qi]
            idx = _topk_indices(scores, min(topk, len(kept_docids)))
            rankings[qid] = [kept_docids[i] for i in idx.tolist()]
            search_t += time.perf_counter() - s0

        m = compute_metrics(rankings, qrels)
        per_dim[str(d)] = {
            "metrics": asdict(m),
            "latency": {
                "encode_queries_s": float(t1 - t0),
                "encode_passages_s": float(t2 - t1),
                "search_total_s": float(search_t),
                "search_avg_ms_per_query": float((search_t / max(1, len(qids))) * 1000.0),
            },
        }

    return {"checkpoint": str(ckpt_dir), "meta": meta, "per_dim": per_dim, "evaluation_mode": "candidate_rerank"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="Path to artifacts/results.json")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    setup_logging(deep_get(cfg, "logging.level", default="INFO"))

    processed = _processed_dir(cfg)
    if not processed.exists():
        raise SystemExit(f"Processed data not found: {processed}. Run `make data` first.")

    dims = [int(x) for x in deep_get(cfg, "eval.dims", required=True)]
    max_queries = deep_get(cfg, "eval.max_queries", default=None)
    max_queries = int(max_queries) if max_queries is not None else None

    qrels = _load_qrels(processed / "qrels_dev.tsv")

    # For the default pipeline we evaluate exact retrieval on the local passages.jsonl.
    max_passages = deep_get(cfg, "eval.max_passages", default=None)
    max_passages = int(max_passages) if max_passages is not None else None
    passages_path = processed / "passages.jsonl"
    passages_size = passages_path.stat().st_size if passages_path.exists() else 0
    passages: Optional[list[tuple[str, str]]] = None
    store = None
    candidates: Optional[dict[str, list[str]]] = None

    # If the corpus is large, prefer candidate reranking if MS MARCO top1000.dev is present.
    large_corpus = passages_size > 2_000_000_000 and max_passages is None
    if not large_corpus:
        passages = _load_passages(processed, max_passages=max_passages)
    else:
        store = open_passage_store(processed, prefer_sqlite=True)
        raw_dir = Path(deep_get(cfg, "data.msmarco.raw_dir", default="data/raw/msmarco"))
        top1000 = raw_dir / "top1000.dev"
        if top1000.exists():
            candidates = _parse_top1000_dev(top1000)

    # Load queries, filtering to those with relevance judgments.
    # For candidate reranking, also filter to queries with candidates.
    all_queries = _load_queries(processed / "queries_dev.jsonl", max_queries=None)
    candidate_qids = set(candidates.keys()) if candidates else None
    judged_queries = [
        (qid, text) for qid, text in all_queries
        if qid in qrels and (candidate_qids is None or qid in candidate_qids)
    ]
    if max_queries is not None:
        judged_queries = judged_queries[:max_queries]
    queries = judged_queries
    log.info("Evaluating %d judged queries (out of %d total dev queries)", len(queries), len(all_queries))

    device = torch.device("cuda" if torch.cuda.is_available() and deep_get(cfg, "train.device", default="auto") in ("auto", "cuda") else "cpu")

    results: dict = {
        "run_name": str(deep_get(cfg, "run_name", default="run")),
        "dataset": str(deep_get(cfg, "data.dataset", required=True)),
        "timestamp_unix": int(time.time()),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": getattr(torch, "__version__", "unknown"),
            "cuda_available": bool(torch.cuda.is_available()),
            "transformers": __import__("transformers").__version__,
        },
        "dims": dims,
        "experiments": {},
    }

    # Baseline: BM25 retrieval.
    if bool(deep_get(cfg, "eval.bm25.enabled", default=True)):
        if passages is None:
            log.warning("BM25 baseline skipped: corpus too large for pure-Python BM25 in this pipeline.")
        else:
            topk = max(max([int(x) for x in deep_get(cfg, "eval.topk", default=[10, 50, 100])]), 1000)
            _, m = bm25_rank(passages=passages, queries=queries, qrels=qrels, topk=topk)
            results["experiments"]["bm25"] = {"metrics": asdict(m)}

    run_name = str(deep_get(cfg, "run_name", default="run"))
    ckpt_root = Path(deep_get(cfg, "paths.checkpoints_dir", default="checkpoints")) / run_name

    # Dense retriever baseline + Matryoshka model. The ablation is "remove Matryoshka loss", i.e., dense_baseline.
    topk = max(max([int(x) for x in deep_get(cfg, "eval.topk", default=[10, 50, 100])]), 1000)
    for exp in ["dense_baseline", "matryoshka"]:
        ckpt_dir = ckpt_root / exp
        if not ckpt_dir.exists():
            raise SystemExit(f"Missing checkpoint for {exp}: {ckpt_dir}. Run `make train` first.")

        if passages is not None:
            results["experiments"][exp] = evaluate_dense_exact(
                ckpt_dir=ckpt_dir,
                passages=passages,
                queries=queries,
                qrels=qrels,
                dims=dims,
                device=device,
                max_length_query=int(deep_get(cfg, "train.max_length_query", default=32)),
                max_length_passage=int(deep_get(cfg, "train.max_length_passage", default=128)),
                batch_size=int(deep_get(cfg, "eval.batch_size", default=32)),
                topk=topk,
            )
        else:
            if candidates is None:
                raise SystemExit(
                    "Corpus is too large for exact evaluation without pre-encoded embeddings.\n"
                    "Expected MS MARCO candidates (top1000.dev) for candidate reranking, but file not found.\n"
                    "Either:\n"
                    "1) Ensure MS MARCO raw data is downloaded/extracted (make data), or\n"
                    "2) Implement full-corpus indexing + ANN search (FAISS) and wire it into evaluation."
                )
            results["experiments"][exp] = evaluate_dense_candidates(
                ckpt_dir=ckpt_dir,
                store=store,
                candidates=candidates,
                queries=queries,
                qrels=qrels,
                dims=dims,
                device=device,
                max_length_query=int(deep_get(cfg, "train.max_length_query", default=32)),
                max_length_passage=int(deep_get(cfg, "train.max_length_passage", default=128)),
                batch_size=int(deep_get(cfg, "eval.batch_size", default=32)),
                topk=topk,
            )

    write_json(args.out, results)
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
