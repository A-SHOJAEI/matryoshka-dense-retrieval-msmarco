from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

from tqdm import tqdm

from src.data.io import read_jsonl

log = logging.getLogger(__name__)


def build_sqlite_from_passages_jsonl(passages_jsonl: str | Path, out_db: str | Path, *, batch_size: int = 10_000) -> None:
    passages_jsonl = Path(passages_jsonl)
    out_db = Path(out_db)
    out_db.parent.mkdir(parents=True, exist_ok=True)

    if out_db.exists():
        raise FileExistsError(f"Refusing to overwrite existing DB: {out_db}")

    conn = sqlite3.connect(str(out_db))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("CREATE TABLE passages (docid INTEGER PRIMARY KEY, text TEXT NOT NULL);")
    conn.execute("CREATE INDEX idx_docid ON passages(docid);")

    buf: list[tuple[int, str]] = []
    inserted = 0
    for row in tqdm(read_jsonl(passages_jsonl), desc="passages.jsonl"):
        docid = int(row["docid"])
        text = str(row["text"])
        buf.append((docid, text))
        if len(buf) >= batch_size:
            conn.executemany("INSERT INTO passages(docid, text) VALUES (?, ?)", buf)
            conn.commit()
            inserted += len(buf)
            buf.clear()
    if buf:
        conn.executemany("INSERT INTO passages(docid, text) VALUES (?, ?)", buf)
        conn.commit()
        inserted += len(buf)
        buf.clear()

    conn.execute("ANALYZE;")
    conn.commit()
    conn.close()
    log.info("Built %s with %d passages", out_db, inserted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--passages_jsonl", required=True)
    parser.add_argument("--out_db", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    build_sqlite_from_passages_jsonl(args.passages_jsonl, args.out_db)


if __name__ == "__main__":
    main()

