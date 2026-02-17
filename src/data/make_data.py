from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data.download_msmarco import download_msmarco
from src.data.make_toy import ToySpec, make_toy_dataset
from src.data.prepare_msmarco import prepare_msmarco
from src.utils.config import deep_get, ensure_dir, load_yaml
from src.utils.logging import setup_logging

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    setup_logging(deep_get(cfg, "logging.level", default="INFO"))

    dataset = deep_get(cfg, "data.dataset", required=True)
    data_dir = Path(deep_get(cfg, "paths.data_dir", default="data"))

    if dataset == "toy":
        out_dir = ensure_dir(data_dir / "processed" / "toy")
        spec = ToySpec(
            num_passages=int(deep_get(cfg, "data.toy.num_passages", required=True)),
            num_queries_train=int(deep_get(cfg, "data.toy.num_queries_train", required=True)),
            num_queries_dev=int(deep_get(cfg, "data.toy.num_queries_dev", required=True)),
            positives_per_query=int(deep_get(cfg, "data.toy.positives_per_query", default=1)),
            seed=int(deep_get(cfg, "seed", default=123)),
        )
        make_toy_dataset(out_dir, spec)
        return

    if dataset == "msmarco":
        raw_dir = Path(deep_get(cfg, "data.msmarco.raw_dir", default=str(data_dir / "raw" / "msmarco")))
        processed_dir = Path(
            deep_get(cfg, "data.msmarco.processed_dir", default=str(data_dir / "processed" / "msmarco"))
        )
        sha256_by_url = deep_get(cfg, "data.msmarco.download.sha256", default={}) or {}
        if not isinstance(sha256_by_url, dict):
            raise SystemExit("data.msmarco.download.sha256 must be a mapping of URL->sha256")

        ensure_dir(raw_dir)
        ensure_dir(processed_dir)
        download_msmarco(raw_dir, sha256_by_url=sha256_by_url)
        prepare_msmarco(raw_dir, processed_dir)
        return

    raise SystemExit(f"Unknown dataset: {dataset}")


if __name__ == "__main__":
    main()

