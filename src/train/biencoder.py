from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.data.io import read_jsonl, read_qrels_tsv
from src.data.stores import PassageStore, open_passage_store
from src.model.encoder import BiEncoder, EncoderConfig, save_biencoder
from src.model.matryoshka import MatryoshkaInfoNCELoss, MatryoshkaLossConfig
from src.utils.config import deep_get, ensure_dir, load_yaml
from src.utils.logging import setup_logging
from src.utils.seed import DeterminismConfig, set_global_seed

log = logging.getLogger(__name__)


def _ddp_info() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size, rank, local_rank


def _init_distributed() -> tuple[bool, int]:
    world_size, rank, local_rank = _ddp_info()
    if world_size <= 1:
        return False, 0
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _get_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def _processed_dir(cfg: dict) -> Path:
    dataset = deep_get(cfg, "data.dataset", required=True)
    if dataset == "toy":
        return Path(deep_get(cfg, "paths.data_dir", default="data")) / "processed" / "toy"
    if dataset == "msmarco":
        return Path(deep_get(cfg, "data.msmarco.processed_dir", required=True))
    raise ValueError(f"Unknown dataset: {dataset}")


def _load_queries(path: Path) -> dict[str, str]:
    m: dict[str, str] = {}
    for row in read_jsonl(path):
        m[str(row["qid"])] = str(row["text"])
    return m


@dataclass(frozen=True)
class TrainExample:
    qid: str
    query: str
    docid: int
    passage: str


class PairDataset(Dataset):
    def __init__(self, examples: list[TrainExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TrainExample:
        return self.examples[idx]


class Collator:
    def __init__(self, tokenizer, *, max_length_query: int, max_length_passage: int):
        self.tok = tokenizer
        self.max_q = int(max_length_query)
        self.max_p = int(max_length_passage)

    def __call__(self, batch: list[TrainExample]) -> dict[str, torch.Tensor]:
        qs = [b.query for b in batch]
        ps = [b.passage for b in batch]
        q = self.tok(
            qs,
            padding=True,
            truncation=True,
            max_length=self.max_q,
            return_tensors="pt",
        )
        p = self.tok(
            ps,
            padding=True,
            truncation=True,
            max_length=self.max_p,
            return_tensors="pt",
        )
        out = {
            "q_input_ids": q["input_ids"],
            "q_attention_mask": q["attention_mask"],
            "p_input_ids": p["input_ids"],
            "p_attention_mask": p["attention_mask"],
        }
        if "token_type_ids" in q:
            out["q_token_type_ids"] = q["token_type_ids"]
        if "token_type_ids" in p:
            out["p_token_type_ids"] = p["token_type_ids"]
        return out


def _build_train_examples(processed: Path, store: PassageStore, *, max_examples: Optional[int]) -> list[TrainExample]:
    queries = _load_queries(processed / "queries_train.jsonl")
    qrels = read_qrels_tsv(processed / "qrels_train.tsv")

    ex: list[TrainExample] = []
    for qid, docid_s, rel in qrels:
        if rel <= 0:
            continue
        if qid not in queries:
            continue
        docid = int(docid_s)
        p = store.get(docid)
        if p is None:
            continue
        ex.append(TrainExample(qid=qid, query=queries[qid], docid=docid, passage=p.text))
        if max_examples is not None and len(ex) >= max_examples:
            break

    if not ex:
        raise RuntimeError(f"No training examples built from {processed}")
    return ex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment", required=True, help="Experiment key under config.experiments.*")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    setup_logging(deep_get(cfg, "logging.level", default="INFO"))

    exp = args.experiment
    exp_cfg = deep_get(cfg, f"experiments.{exp}", required=True)

    # Reproducibility controls.
    det = DeterminismConfig(
        seed=int(deep_get(cfg, "seed", default=123)),
        deterministic=bool(deep_get(cfg, "train.deterministic", default=False)),
        warn_only=bool(deep_get(cfg, "train.deterministic_warn_only", default=True)),
    )
    set_global_seed(det)

    distributed, rank = _init_distributed()

    device = _get_device(str(deep_get(cfg, "train.device", default="auto")))
    if distributed and torch.cuda.is_available():
        _, _, local_rank = _ddp_info()
        device = torch.device(f"cuda:{local_rank}")

    processed = _processed_dir(cfg)
    if not processed.exists():
        raise SystemExit(f"Processed data not found: {processed}. Run `make data` first.")

    # Use sqlite passage store if present (required for large corpora).
    store = open_passage_store(processed, prefer_sqlite=True)

    max_examples = deep_get(cfg, "train.max_train_examples", default=None)
    if max_examples is not None:
        max_examples = int(max_examples)

    examples = _build_train_examples(processed, store, max_examples=max_examples)

    tok = AutoTokenizer.from_pretrained(str(deep_get(cfg, "model.hf_model_name", required=True)))
    model_cfg = EncoderConfig(
        hf_model_name=str(deep_get(cfg, "model.hf_model_name", required=True)),
        pooling=str(deep_get(cfg, "model.pooling", default="mean")),
        projection_dim=int(deep_get(cfg, "model.projection_dim", default=768)),
        l2_normalize=bool(deep_get(cfg, "model.l2_normalize", default=True)),
    )
    model = BiEncoder(model_cfg).to(device)

    loss_cfg = MatryoshkaLossConfig(
        dims=[int(x) for x in deep_get(exp_cfg, "matryoshka_dims", required=True)],
        temperature=float(deep_get(exp_cfg, "temperature", default=0.05)),
        use_matryoshka=bool(deep_get(exp_cfg, "use_matryoshka", default=False)),
    )
    criterion = MatryoshkaInfoNCELoss(loss_cfg)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )

    ds = PairDataset(examples)
    sampler = None
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=True, seed=det.seed)

    dl = DataLoader(
        ds,
        batch_size=int(deep_get(cfg, "train.batch_size", default=32)),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(deep_get(cfg, "train.num_workers", default=0)),
        collate_fn=Collator(
            tok,
            max_length_query=int(deep_get(cfg, "train.max_length_query", default=32)),
            max_length_passage=int(deep_get(cfg, "train.max_length_passage", default=128)),
        ),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    lr = float(deep_get(cfg, "train.lr", default=2e-5))
    wd = float(deep_get(cfg, "train.weight_decay", default=0.01))
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    fp16 = bool(deep_get(cfg, "train.fp16", default=False))
    scaler = torch.cuda.amp.GradScaler(enabled=fp16 and device.type == "cuda")

    epochs = int(deep_get(cfg, "train.epochs", default=1))
    max_steps = deep_get(cfg, "train.max_steps", default=None)
    max_steps = int(max_steps) if max_steps is not None else None

    log_every = int(deep_get(cfg, "train.log_every", default=50))
    save_every = int(deep_get(cfg, "train.save_every", default=500))

    run_name = str(deep_get(cfg, "run_name", default="run"))
    ckpt_root = Path(deep_get(cfg, "paths.checkpoints_dir", default="checkpoints")) / run_name / exp
    if _is_main_process():
        ensure_dir(ckpt_root)
        (ckpt_root / "config.json").write_text(
            json.dumps(cfg, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8"
        )

    step = 0
    model.train()
    start = time.time()
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in dl:
            step += 1
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=fp16 and device.type == "cuda"):
                q_tt = batch.get("q_token_type_ids", None)
                p_tt = batch.get("p_token_type_ids", None)
                if distributed:
                    q = model.module.encode_queries(batch["q_input_ids"], batch["q_attention_mask"], q_tt)
                    p = model.module.encode_passages(batch["p_input_ids"], batch["p_attention_mask"], p_tt)
                else:
                    q = model.encode_queries(batch["q_input_ids"], batch["q_attention_mask"], q_tt)
                    p = model.encode_passages(batch["p_input_ids"], batch["p_attention_mask"], p_tt)
                loss = criterion(q, p)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            if _is_main_process() and (step % log_every == 0):
                elapsed = time.time() - start
                log.info(
                    "exp=%s epoch=%d step=%d loss=%.4f (%.2fs)",
                    exp,
                    epoch,
                    step,
                    float(loss.detach().cpu()),
                    elapsed,
                )

            if _is_main_process() and (step % save_every == 0):
                # Unwrap DDP.
                m = model.module if distributed else model
                save_biencoder(
                    str(ckpt_root),
                    m,
                    config={
                        "run_name": run_name,
                        "experiment": exp,
                        "seed": det.seed,
                        "model": {
                            "hf_model_name": model_cfg.hf_model_name,
                            "pooling": model_cfg.pooling,
                            "projection_dim": model_cfg.projection_dim,
                            "l2_normalize": model_cfg.l2_normalize,
                        },
                        "loss": {
                            "use_matryoshka": loss_cfg.use_matryoshka,
                            "dims": loss_cfg.dims,
                            "temperature": loss_cfg.temperature,
                        },
                    },
                    tokenizer=tok,
                )

            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break

    if _is_main_process():
        m = model.module if distributed else model
        save_biencoder(
            str(ckpt_root),
            m,
            config={
                "run_name": run_name,
                "experiment": exp,
                "seed": det.seed,
                "model": {
                    "hf_model_name": model_cfg.hf_model_name,
                    "pooling": model_cfg.pooling,
                    "projection_dim": model_cfg.projection_dim,
                    "l2_normalize": model_cfg.l2_normalize,
                },
                "loss": {
                    "use_matryoshka": loss_cfg.use_matryoshka,
                    "dims": loss_cfg.dims,
                    "temperature": loss_cfg.temperature,
                },
            },
            tokenizer=tok,
        )
        log.info("Saved checkpoint to %s", ckpt_root)

    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
