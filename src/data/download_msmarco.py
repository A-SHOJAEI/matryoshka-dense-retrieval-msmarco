from __future__ import annotations

import argparse
import logging
import os
import tarfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

from src.utils.hashing import sha256_file

log = logging.getLogger(__name__)


MSMARCO_URLS = [
    "https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz",
    "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.train.tsv",
    "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv",
    "https://msmarco.z22.web.core.windows.net/msmarcoranking/top1000.dev.tar.gz",
]


def _download(url: str, out_path: Path, *, expected_sha256: Optional[str], timeout_s: int = 60) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Fast path: already present and verified.
    if out_path.exists():
        if expected_sha256:
            got = sha256_file(out_path)
            if got.lower() == expected_sha256.lower():
                log.info("OK (sha256) %s", out_path.name)
                return
            raise RuntimeError(
                f"Checksum mismatch for existing file {out_path}.\n"
                f"expected sha256={expected_sha256}\n"
                f"got      sha256={got}\n"
                "Delete the file and retry."
            )
        log.info("Exists (no checksum provided): %s", out_path.name)
        return

    log.info("Downloading %s -> %s", url, out_path)
    with requests.get(url, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        tmp = out_path.with_suffix(out_path.suffix + ".partial")
        h = None
        if expected_sha256:
            import hashlib

            h = hashlib.sha256()

        with tmp.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                if h is not None:
                    h.update(chunk)
                pbar.update(len(chunk))

    os.replace(tmp, out_path)
    if expected_sha256:
        got = h.hexdigest() if h is not None else sha256_file(out_path)
        if got.lower() != expected_sha256.lower():
            raise RuntimeError(
                f"Checksum mismatch for {out_path}.\nexpected sha256={expected_sha256}\n"
                f"got      sha256={got}"
            )
        log.info("Verified sha256 for %s", out_path.name)
    else:
        log.warning("No checksum provided for %s; skipped checksum verification.", out_path.name)


def _extract_tar_gz(path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Extracting %s -> %s", path.name, out_dir)
    with tarfile.open(path, "r:gz") as tf:
        tf.extractall(out_dir)


def download_msmarco(out_dir: str | Path, *, sha256_by_url: dict[str, str] | None = None) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sha256_by_url = sha256_by_url or {}

    downloaded: dict[str, Path] = {}
    for url in MSMARCO_URLS:
        fname = url.split("/")[-1]
        out_path = out_dir / fname
        _download(url, out_path, expected_sha256=sha256_by_url.get(url))
        downloaded[url] = out_path

    # Extract archives next to downloads for a predictable layout.
    _extract_tar_gz(out_dir / "collectionandqueries.tar.gz", out_dir)
    _extract_tar_gz(out_dir / "top1000.dev.tar.gz", out_dir)

    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output directory for raw MS MARCO files.")
    parser.add_argument(
        "--sha256",
        default=None,
        help="Optional path to a YAML/JSON mapping of URL->sha256 for verification.",
    )
    args = parser.parse_args()

    import json
    import yaml

    sha256_by_url = {}
    if args.sha256:
        p = Path(args.sha256)
        if not p.exists():
            raise SystemExit(f"sha256 mapping not found: {p}")
        if p.suffix.lower() in {".yaml", ".yml"}:
            sha256_by_url = yaml.safe_load(p.read_text(encoding="utf-8"))
        else:
            sha256_by_url = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(sha256_by_url, dict):
            raise SystemExit("sha256 mapping must be an object/dict of URL->sha256")

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    download_msmarco(args.out, sha256_by_url=sha256_by_url)


if __name__ == "__main__":
    main()

