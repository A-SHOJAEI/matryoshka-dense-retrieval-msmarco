from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "BM25 baseline using Pyserini/Lucene (MS MARCO scale).\n"
            "This is optional and not required for the default smoke pipeline."
        )
    )
    parser.add_argument("--collection_tsv", required=True, help="Path to MS MARCO collection.tsv")
    parser.add_argument("--queries_tsv", required=True, help="Path to queries.dev.tsv (qid<tab>text)")
    parser.add_argument("--out", required=True, help="Output run file path (TSV/JSON).")
    args = parser.parse_args()

    try:
        from pyserini.search.lucene import LuceneSearcher  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Pyserini is not installed or not working in this environment.\n"
            "Install it in your venv and ensure Java is available.\n"
            f"Original import error: {e}"
        )

    # This script is intentionally minimal: Pyserini indexing is typically done via CLI
    # for MS MARCO scale (and can take hours). We provide this file as a placeholder for
    # users who want Pyserini-based BM25 and are comfortable configuring Lucene indexes.
    raise SystemExit(
        "Pyserini BM25 indexing/search is environment-specific (Java, disk, memory).\n"
        "Use Pyserini's official MS MARCO instructions to build a Lucene index, then adapt this script "
        "to point LuceneSearcher at that index.\n"
        "The default repository baseline (smoke) uses rank-bm25 for small corpora."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    main()

