from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class MatryoshkaLossConfig:
    dims: list[int]
    temperature: float
    use_matryoshka: bool


class MatryoshkaInfoNCELoss(nn.Module):
    """
    In-batch InfoNCE computed at multiple prefix embedding dimensions.

    Given query and positive passage embeddings [B, D], for each dim d:
    - truncate to [:d]
    - (re)normalize
    - compute similarity logits and cross-entropy against diagonal targets
    The loss is averaged across dims.
    """

    def __init__(self, cfg: MatryoshkaLossConfig):
        super().__init__()
        if not cfg.dims:
            raise ValueError("dims must be non-empty")
        self.cfg = cfg

    @staticmethod
    def _info_nce(q: torch.Tensor, p: torch.Tensor, temperature: float) -> torch.Tensor:
        # q,p: [B, d], assumed normalized.
        logits = (q @ p.t()) / float(temperature)
        targets = torch.arange(q.size(0), device=q.device)
        return torch.nn.functional.cross_entropy(logits, targets)

    def forward(self, q_emb: torch.Tensor, p_emb: torch.Tensor) -> torch.Tensor:
        if q_emb.ndim != 2 or p_emb.ndim != 2:
            raise ValueError("Expected 2D embeddings [B, D]")
        if q_emb.shape != p_emb.shape:
            raise ValueError(f"Shape mismatch: q={tuple(q_emb.shape)} p={tuple(p_emb.shape)}")
        D = q_emb.size(1)

        dims = self.cfg.dims if self.cfg.use_matryoshka else [max(self.cfg.dims)]
        losses = []
        for d in dims:
            if d > D:
                raise ValueError(f"Configured dim {d} exceeds embedding dim {D}")
            q = torch.nn.functional.normalize(q_emb[:, :d], p=2, dim=-1)
            p = torch.nn.functional.normalize(p_emb[:, :d], p=2, dim=-1)
            losses.append(self._info_nce(q, p, self.cfg.temperature))
        return torch.stack(losses).mean()

