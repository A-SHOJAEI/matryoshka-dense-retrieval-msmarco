from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data.io import read_jsonl
from src.model.encoder import load_biencoder

log = logging.getLogger(__name__)


@torch.no_grad()
def encode_passages(
    *,
    ckpt_dir: str | Path,
    passages_jsonl: str | Path,
    out_dir: str | Path,
    batch_size: int = 128,
    max_length: int = 192,
    shard_size: int = 200_000,
    device: str = "auto",
) -> None:
    ckpt_dir = Path(ckpt_dir)
    passages_jsonl = Path(passages_jsonl)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device))
    model, meta = load_biencoder(str(ckpt_dir), map_location=dev)
    model = model.to(dev).eval()
    tok = (
        AutoTokenizer.from_pretrained(str(ckpt_dir / "tokenizer"))
        if (ckpt_dir / "tokenizer").exists()
        else AutoTokenizer.from_pretrained(meta["model"]["hf_model_name"])
    )

    shard_docids: list[int] = []
    shard_vecs: list[np.ndarray] = []
    shard_idx = 0

    def flush() -> None:
        nonlocal shard_idx
        if not shard_docids:
            return
        docids_arr = np.array(shard_docids, dtype=np.int64)
        vecs_arr = np.concatenate(shard_vecs, axis=0).astype(np.float32, copy=False)
        np.save(out_dir / f"docids.{shard_idx:04d}.npy", docids_arr)
        np.save(out_dir / f"emb.{shard_idx:04d}.npy", vecs_arr)
        shard_docids.clear()
        shard_vecs.clear()
        shard_idx += 1

    batch_docids: list[int] = []
    batch_texts: list[str] = []
    for row in tqdm(read_jsonl(passages_jsonl), desc="encode passages"):
        batch_docids.append(int(row["docid"]))
        batch_texts.append(str(row["text"]))
        if len(batch_texts) >= batch_size:
            toks = tok(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            toks = {k: v.to(dev) for k, v in toks.items()}
            emb = model.encoder(**toks).detach().float().cpu().numpy()
            shard_docids.extend(batch_docids)
            shard_vecs.append(emb)
            batch_docids.clear()
            batch_texts.clear()
            if len(shard_docids) >= shard_size:
                flush()

    if batch_texts:
        toks = tok(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        toks = {k: v.to(dev) for k, v in toks.items()}
        emb = model.encoder(**toks).detach().float().cpu().numpy()
        shard_docids.extend(batch_docids)
        shard_vecs.append(emb)
        batch_docids.clear()
        batch_texts.clear()

    flush()
    log.info("Wrote passage embedding shards to %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Checkpoint dir (e.g., checkpoints/msmarco/matryoshka)")
    parser.add_argument("--passages", required=True, help="Processed passages.jsonl")
    parser.add_argument("--out", required=True, help="Output directory for embedding shards")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--shard_size", type=int, default=200_000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    encode_passages(
        ckpt_dir=args.ckpt,
        passages_jsonl=args.passages,
        out_dir=args.out,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shard_size=args.shard_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()

