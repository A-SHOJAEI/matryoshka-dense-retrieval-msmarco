from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import nn
from transformers import AutoModel

log = logging.getLogger(__name__)


Pooling = Literal["mean"]


@dataclass(frozen=True)
class EncoderConfig:
    hf_model_name: str
    pooling: Pooling
    projection_dim: int
    l2_normalize: bool


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden: [B, T, H], attention_mask: [B, T]
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # [B, T, 1]
    summed = (last_hidden * mask).sum(dim=1)  # [B, H]
    denom = mask.sum(dim=1).clamp_min(1.0)  # [B, 1]
    return summed / denom


class TransformerEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = AutoModel.from_pretrained(cfg.hf_model_name)
        hidden = int(getattr(self.backbone.config, "hidden_size"))
        self.proj = nn.Linear(hidden, int(cfg.projection_dim), bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "return_dict": True}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.backbone(**kwargs)
        last_hidden = out.last_hidden_state
        if self.cfg.pooling == "mean":
            pooled = mean_pool(last_hidden, attention_mask)
        else:
            raise ValueError(f"Unknown pooling: {self.cfg.pooling}")
        emb = self.proj(pooled)
        if self.cfg.l2_normalize:
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        return emb


class BiEncoder(nn.Module):
    """
    Shared-weight bi-encoder (query and passage encoders share parameters).
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.encoder = TransformerEncoder(cfg)

    @property
    def projection_dim(self) -> int:
        return int(self.encoder.cfg.projection_dim)

    def encode_queries(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    def encode_passages(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)


def save_biencoder(ckpt_dir: str, model: BiEncoder, *, config: dict, tokenizer: Optional[object] = None) -> None:
    from pathlib import Path
    import json

    p = Path(ckpt_dir)
    p.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config}, p / "model.pt")
    (p / "meta.json").write_text(json.dumps(config, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(p / "tokenizer"))


def load_biencoder(ckpt_dir: str, *, map_location: str | torch.device = "cpu") -> tuple[BiEncoder, dict]:
    from pathlib import Path

    p = Path(ckpt_dir)
    blob = torch.load(p / "model.pt", map_location=map_location)
    cfg = blob["config"]
    enc_cfg = EncoderConfig(
        hf_model_name=str(cfg["model"]["hf_model_name"]),
        pooling=str(cfg["model"].get("pooling", "mean")),
        projection_dim=int(cfg["model"]["projection_dim"]),
        l2_normalize=bool(cfg["model"].get("l2_normalize", True)),
    )
    model = BiEncoder(enc_cfg)
    model.load_state_dict(blob["model_state"])
    return model, cfg
