"""The per-platform YAML (``config/zollo/<platform>.yaml``) both scripts share."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .identify import IdentificationConfig


@dataclass
class PlatformConfig:
    path: Path
    platform: str
    dataset: Path
    splits: tuple[str, ...]
    identification: IdentificationConfig
    evaluation: dict[str, Any]
    references: dict[str, Any]
    output_dir: Path
    repo: Path

    def resolve(self, value: str | Path) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else self.repo / candidate

    def real_files(self) -> dict[str, Path]:
        return {name: self.resolve(path) for name, path in (self.evaluation.get("real") or {}).items()}


def load_platform(path: str | Path, repo: str | Path) -> PlatformConfig:
    repo = Path(repo)
    path = Path(path)
    path = path if path.is_absolute() else repo / path
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    known = {"platform", "dataset", "splits", "identification", "preprocessing", "evaluation", "references",
             "output_dir"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    identification = IdentificationConfig.from_mapping(raw.get("identification") or {}, raw.get("preprocessing") or {})
    out = PlatformConfig(path, str(raw["platform"]), Path(), tuple(raw.get("splits") or ("train",)), identification,
                         dict(raw.get("evaluation") or {}), dict(raw.get("references") or {}), Path(), repo)
    out.dataset = out.resolve(raw["dataset"])
    out.output_dir = out.resolve(raw.get("output_dir") or f"reports/zollo/{out.platform}")
    return out


@dataclass
class FairConfig:
    """``config/zollo/fair/<platform>.yaml``: the fair baselines (03_fair_comparison_plan.md)."""

    path: Path
    platform: str
    dataset: Path
    asset: str
    selection: dict[str, Any]
    output_error: dict[str, Any]
    percentile: dict[str, Any]
    evaluation: dict[str, Any]
    references: dict[str, Any]
    lnn_pipeline: dict[str, Any]
    output_dir: Path
    repo: Path

    def resolve(self, value: str | Path) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else self.repo / candidate

    def real_files(self) -> dict[str, Path]:
        return {name: self.resolve(path) for name, path in (self.evaluation.get("real") or {}).items()}


def load_fair_config(path: str | Path, repo: str | Path) -> FairConfig:
    repo = Path(repo)
    path = Path(path)
    path = path if path.is_absolute() else repo / path
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    known = {"platform", "dataset", "asset", "selection", "output_error", "percentile", "evaluation", "references",
             "lnn_pipeline", "output_dir"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    out = FairConfig(path, str(raw["platform"]), Path(), str(raw["asset"]), dict(raw.get("selection") or {}),
                     dict(raw.get("output_error") or {}), dict(raw.get("percentile") or {}),
                     dict(raw.get("evaluation") or {}), dict(raw.get("references") or {}),
                     dict(raw.get("lnn_pipeline") or {}), Path(), repo)
    out.dataset = out.resolve(raw["dataset"])
    out.output_dir = out.resolve(raw.get("output_dir") or f"reports/zollo_fair/{out.platform}")
    return out
