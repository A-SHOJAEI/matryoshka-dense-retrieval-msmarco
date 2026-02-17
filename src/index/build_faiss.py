from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def build_faiss_flat_ip(*, emb_dir: str | Path, dims: list[int], out_dir: str | Path) -> None:
    """
    Build FAISS IndexFlatIP indexes per dimension from embedding shards.

    Requires `faiss` to be installed (not included in core requirements, since wheels are platform-dependent).
    """
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise SystemExit(
            "faiss is not installed.\n"
            "Install faiss-cpu (or faiss-gpu) into your venv, then retry.\n"
            f"Original import error: {e}"
        )

    emb_dir = Path(emb_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_files = sorted(emb_dir.glob("emb.*.npy"))
    if not emb_files:
        raise FileNotFoundError(f"No embedding shards found at {emb_dir} (expected emb.0000.npy, ...)")

    for d in dims:
        index = faiss.IndexFlatIP(d)
        for ef in emb_files:
            x = np.load(ef).astype(np.float32, copy=False)
            if x.shape[1] < d:
                raise ValueError(f"Shard {ef} has dim {x.shape[1]} < requested {d}")
            # Truncate; embeddings are assumed normalized at full dim, so renormalize per dim.
            y = x[:, :d]
            n = np.linalg.norm(y, axis=1, keepdims=True)
            y = y / np.maximum(n, 1e-12)
            index.add(y)
        faiss.write_index(index, str(out_dir / f"flatip.d{d}.faiss"))
        log.info("Wrote %s", out_dir / f"flatip.d{d}.faiss")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--dims", nargs="+", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    build_faiss_flat_ip(emb_dir=args.emb_dir, dims=args.dims, out_dir=args.out)


if __name__ == "__main__":
    main()

