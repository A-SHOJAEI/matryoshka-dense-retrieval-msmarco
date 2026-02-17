from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ConfigError(f"Top-level config must be a mapping: {p}")
    return obj


def deep_get(d: dict[str, Any], key_path: str, *, default: Any = None, required: bool = False) -> Any:
    cur: Any = d
    parts = key_path.split(".")
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            if required:
                raise ConfigError(f"Missing required config key: {key_path}")
            return default
        cur = cur[part]
    return cur


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(frozen=True)
class Paths:
    work_dir: Path
    data_dir: Path
    checkpoints_dir: Path


def parse_paths(cfg: dict[str, Any]) -> Paths:
    work_dir = Path(deep_get(cfg, "paths.work_dir", default=".", required=False))
    data_dir = Path(deep_get(cfg, "paths.data_dir", default="data", required=False))
    ckpt_dir = Path(deep_get(cfg, "paths.checkpoints_dir", default="checkpoints", required=False))
    return Paths(work_dir=work_dir, data_dir=data_dir, checkpoints_dir=ckpt_dir)

