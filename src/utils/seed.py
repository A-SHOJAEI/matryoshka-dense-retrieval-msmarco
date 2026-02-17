from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeterminismConfig:
    seed: int
    deterministic: bool
    warn_only: bool


def set_global_seed(cfg: DeterminismConfig) -> None:
    """
    Best-effort reproducibility controls.

    Notes:
    - Deterministic algorithms can reduce throughput and may raise on some CUDA ops.
    - We set environment variables early where possible.
    """
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    try:
        import torch

        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(cfg.deterministic)
        if cfg.deterministic:
            # Matmul determinism (may be required for some CUDA versions).
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=bool(cfg.warn_only))
    except Exception:
        # Keep seed setting usable without torch installed (e.g., docs tooling).
        return

