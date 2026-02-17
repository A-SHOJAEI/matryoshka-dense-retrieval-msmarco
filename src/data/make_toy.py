from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from src.utils.config import ensure_dir

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToySpec:
    num_passages: int
    num_queries_train: int
    num_queries_dev: int
    positives_per_query: int
    seed: int


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def _write_tsv(path: Path, rows: list[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for qid, docid, rel in rows:
            f.write(f"{qid}\t{docid}\t{rel}\n")


def make_toy_dataset(out_dir: str | Path, spec: ToySpec) -> dict[str, Path]:
    """
    Create a tiny IR dataset with:
    - passages.jsonl: {"docid": int, "text": str}
    - queries_train.jsonl / queries_dev.jsonl: {"qid": str, "text": str}
    - qrels_train.tsv / qrels_dev.tsv: qid <tab> docid <tab> rel
    """
    out_dir = ensure_dir(out_dir)
    rng = random.Random(spec.seed)

    topics = [
        "astronomy",
        "biology",
        "computer science",
        "economics",
        "geography",
        "history",
        "literature",
        "mathematics",
        "medicine",
        "music",
        "physics",
        "sports",
    ]
    keywords = {
        "astronomy": ["planet", "star", "galaxy", "telescope", "orbit"],
        "biology": ["cell", "gene", "protein", "evolution", "enzyme"],
        "computer science": ["algorithm", "database", "network", "compiler", "python"],
        "economics": ["inflation", "market", "supply", "demand", "gdp"],
        "geography": ["river", "mountain", "desert", "latitude", "country"],
        "history": ["empire", "revolution", "treaty", "dynasty", "war"],
        "literature": ["novel", "poem", "author", "metaphor", "plot"],
        "mathematics": ["theorem", "proof", "matrix", "integral", "prime"],
        "medicine": ["diagnosis", "therapy", "vaccine", "symptom", "disease"],
        "music": ["melody", "harmony", "rhythm", "composer", "orchestra"],
        "physics": ["energy", "force", "quantum", "relativity", "particle"],
        "sports": ["tournament", "coach", "score", "athlete", "league"],
    }

    passages: list[dict] = []
    for docid in range(spec.num_passages):
        t = rng.choice(topics)
        k = rng.sample(keywords[t], k=3)
        text = (
            f"This passage is about {t}. "
            f"It mentions {k[0]}, {k[1]}, and {k[2]}. "
            "It is written to support retrieval experiments."
        )
        passages.append({"docid": docid, "text": text, "topic": t})

    def make_queries(n: int, offset: int) -> tuple[list[dict], list[tuple[str, str, int]]]:
        qs: list[dict] = []
        qrels: list[tuple[str, str, int]] = []
        for i in range(n):
            qid = f"q{offset + i}"
            # Pick a target passage and form a query with overlapping keywords.
            pos = rng.randrange(spec.num_passages)
            topic = passages[pos]["topic"]
            kw = rng.sample(keywords[topic], k=2)
            qtext = f"{topic} {kw[0]} {kw[1]}"
            qs.append({"qid": qid, "text": qtext})
            qrels.append((qid, str(pos), 1))
        return qs, qrels

    queries_train, qrels_train = make_queries(spec.num_queries_train, 0)
    queries_dev, qrels_dev = make_queries(spec.num_queries_dev, spec.num_queries_train)

    out = {
        "passages": out_dir / "passages.jsonl",
        "queries_train": out_dir / "queries_train.jsonl",
        "queries_dev": out_dir / "queries_dev.jsonl",
        "qrels_train": out_dir / "qrels_train.tsv",
        "qrels_dev": out_dir / "qrels_dev.tsv",
    }

    _write_jsonl(out["passages"], [{"docid": p["docid"], "text": p["text"]} for p in passages])
    _write_jsonl(out["queries_train"], queries_train)
    _write_jsonl(out["queries_dev"], queries_dev)
    _write_tsv(out["qrels_train"], qrels_train)
    _write_tsv(out["qrels_dev"], qrels_dev)

    log.info("Wrote toy dataset to %s", out_dir)
    return out

