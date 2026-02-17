from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.data.io import read_jsonl


@dataclass(frozen=True)
class Passage:
    docid: int
    text: str


class PassageStore:
    def get(self, docid: int) -> Optional[Passage]:
        raise NotImplementedError


class InMemoryPassageStore(PassageStore):
    def __init__(self, mapping: dict[int, str]):
        self._m = mapping

    @classmethod
    def from_jsonl(cls, passages_jsonl: str | Path, *, max_items: Optional[int] = None) -> "InMemoryPassageStore":
        m: dict[int, str] = {}
        for i, row in enumerate(read_jsonl(passages_jsonl)):
            if max_items is not None and i >= max_items:
                break
            docid = int(row["docid"])
            m[docid] = str(row["text"])
        return cls(m)

    def get(self, docid: int) -> Optional[Passage]:
        t = self._m.get(int(docid))
        if t is None:
            return None
        return Passage(docid=int(docid), text=t)

    def __len__(self) -> int:
        return len(self._m)


class SqlitePassageStore(PassageStore):
    def __init__(self, db_path: str | Path):
        self._p = Path(db_path)
        self._conn = sqlite3.connect(str(self._p), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")

    def get(self, docid: int) -> Optional[Passage]:
        cur = self._conn.execute("SELECT text FROM passages WHERE docid=?", (int(docid),))
        row = cur.fetchone()
        if row is None:
            return None
        return Passage(docid=int(docid), text=str(row[0]))

    def close(self) -> None:
        self._conn.close()


def open_passage_store(processed_dir: str | Path, *, prefer_sqlite: bool = True) -> PassageStore:
    processed_dir = Path(processed_dir)
    sqlite_path = processed_dir / "passages.sqlite"
    jsonl_path = processed_dir / "passages.jsonl"
    if prefer_sqlite and sqlite_path.exists():
        return SqlitePassageStore(sqlite_path)
    if jsonl_path.exists():
        # MS MARCO is huge; loading passages.jsonl into memory is only intended for toy/smoke and small corpora.
        try:
            if jsonl_path.stat().st_size > 1_000_000_000:
                raise RuntimeError(
                    f"{jsonl_path} is very large; refusing to load into memory.\n"
                    f"Build a SQLite passage store instead:\n"
                    f"  python -m src.data.build_passage_db --passages_jsonl {jsonl_path} --out_db {processed_dir/'passages.sqlite'}"
                )
        except FileNotFoundError:
            pass
        return InMemoryPassageStore.from_jsonl(jsonl_path)
    raise FileNotFoundError(f"No passage store found at {processed_dir} (expected passages.sqlite or passages.jsonl)")
