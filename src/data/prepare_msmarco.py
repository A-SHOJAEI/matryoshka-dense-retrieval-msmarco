from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

log = logging.getLogger(__name__)


_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def _clean_text(s: str) -> str:
    s = _CONTROL_CHARS.sub(" ", s)
    return " ".join(s.split())


def _read_tsv(path: Path) -> Iterable[list[str]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            yield line.split("\t")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def _parse_qrels_row(cols: list[str]) -> tuple[str, str, int]:
    # Support common qrels layouts:
    # - qid docid
    # - qid docid rel
    # - qid 0 docid rel  (TREC-style)
    if len(cols) == 2:
        qid, docid = cols
        rel = 1
    elif len(cols) == 3:
        qid, docid, rel_s = cols
        rel = int(rel_s)
    else:
        qid = cols[0]
        docid = cols[2]
        rel = int(cols[3]) if len(cols) > 3 else 1
    return qid, docid, rel


def prepare_msmarco(raw_dir: str | Path, out_dir: str | Path) -> dict[str, Path]:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    collection = raw_dir / "collection.tsv"
    q_train = raw_dir / "queries.train.tsv"
    q_dev = raw_dir / "queries.dev.tsv"
    qrels_train = raw_dir / "qrels.train.tsv"
    qrels_dev = raw_dir / "qrels.dev.tsv"

    missing = [p for p in [collection, q_train, q_dev, qrels_train, qrels_dev] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing expected MS MARCO files in raw_dir. Missing:\n" + "\n".join(str(p) for p in missing)
        )

    out = {
        "passages": out_dir / "passages.jsonl",
        "queries_train": out_dir / "queries_train.jsonl",
        "queries_dev": out_dir / "queries_dev.jsonl",
        "qrels_train": out_dir / "qrels_train.tsv",
        "qrels_dev": out_dir / "qrels_dev.tsv",
    }

    log.info("Preparing passages: %s", collection)
    def passage_rows() -> Iterable[dict]:
        for cols in tqdm(_read_tsv(collection), desc="collection.tsv"):
            if len(cols) < 2:
                continue
            pid, text = cols[0], cols[1]
            yield {"docid": int(pid), "text": _clean_text(text)}

    _write_jsonl(out["passages"], passage_rows())

    def query_rows(path: Path, desc: str) -> Iterable[dict]:
        for cols in tqdm(_read_tsv(path), desc=desc):
            if len(cols) < 2:
                continue
            qid, text = cols[0], cols[1]
            yield {"qid": str(qid), "text": _clean_text(text)}

    _write_jsonl(out["queries_train"], query_rows(q_train, "queries.train.tsv"))
    _write_jsonl(out["queries_dev"], query_rows(q_dev, "queries.dev.tsv"))

    def write_qrels(src: Path, dst: Path, desc: str) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("w", encoding="utf-8") as f:
            for cols in tqdm(_read_tsv(src), desc=desc):
                qid, docid, rel = _parse_qrels_row(cols)
                f.write(f"{qid}\t{docid}\t{rel}\n")

    write_qrels(qrels_train, out["qrels_train"], "qrels.train.tsv")
    write_qrels(qrels_dev, out["qrels_dev"], "qrels.dev.tsv")

    log.info("Prepared MS MARCO processed data at %s", out_dir)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    prepare_msmarco(args.raw_dir, args.out)


if __name__ == "__main__":
    main()

