"""Estimate the four responsibility-direction scales from training data only."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .responsibility_features import (
    DATASET_BENCHMARKS,
    load_feature_segment,
    read_sampler_index,
)


SCHEMA_VERSION = "deep-rule-direction-scale-v2"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def initial_directions(components: np.ndarray) -> np.ndarray:
    """Return D1..D4 using the declared equal initial horizon weights."""

    value = np.asarray(components, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 8:
        raise ValueError("rule_components must have shape [N, 8]")
    if not np.isfinite(value).all():
        raise FloatingPointError("rule_components contain NaN or Inf")
    c1, c5, c20, c60, e1, e5, e20, e60 = value.T
    return np.column_stack(
        (
            (c5 + c20 + c60) / 3.0,
            -(c1 + c5) / 2.0,
            (e5 + e20 + e60) / 3.0,
            -(e1 + e5) / 2.0,
        )
    )


def estimate_direction_scale(index: Any, components: np.ndarray) -> Dict[str, Any]:
    """Estimate daily cross-section-centered RMS scales without labels."""

    rows = len(index)
    if components.shape != (rows, 8):
        raise ValueError("index and component row count differ")
    if rows == 0:
        raise ValueError("empty training segment")
    dates = np.asarray(index.get_level_values("datetime"))
    boundaries = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1, rows]
    square_sum = np.zeros(4, dtype=np.float64)
    count = 0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        daily_components = np.asarray(components[int(start) : int(end)])
        valid = np.any(daily_components != 0.0, axis=1)
        if not np.any(valid):
            continue
        directions = initial_directions(daily_components[valid])
        centered = directions - directions.mean(axis=0, keepdims=True)
        square_sum += np.square(centered).sum(axis=0, dtype=np.float64)
        count += int(valid.sum())
    if count == 0:
        raise FloatingPointError("training segment contains no valid rule rows")
    rms = np.sqrt(square_sum / float(count))
    if not np.isfinite(rms).all() or np.any(rms <= 1e-12):
        raise FloatingPointError("invalid direction RMS: %r" % rms.tolist())
    return {
        "direction_scale": rms.tolist(),
        "rows": rows,
        "days": int(len(boundaries) - 1),
        "square_sum": square_sum.tolist(),
        "count": count,
    }


def build_direction_scale(
    dataset: str,
    sampler_dir: Path,
    feature_root: Path,
    output: Path,
) -> Dict[str, Any]:
    """Write one new train-only direction-scale JSON file."""

    if dataset not in DATASET_BENCHMARKS:
        raise ValueError("unsupported dataset: %s" % dataset)
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    train_sampler = Path(sampler_dir).expanduser().resolve(strict=True) / "train.pkl"
    index = read_sampler_index(train_sampler, "train")
    segment = load_feature_segment(feature_root, "train", expected_index=index)
    feature_dataset = segment.manifest.get("dataset_id")
    if feature_dataset != dataset:
        raise ValueError(
            "feature dataset mismatch: expected %s, got %s"
            % (dataset, feature_dataset)
        )
    result = estimate_direction_scale(index, np.asarray(segment.rule_components))
    payload: Dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "dataset": dataset,
        "source_segment": "train",
        "development_or_diagnostic": True,
        "created_utc": _utc_now(),
        "formula": {
            "D1": "mean(c_5,c_20,c_60)",
            "D2": "-mean(c_1,c_5)",
            "D3": "mean(e_5,e_20,e_60)",
            "D4": "-mean(e_1,e_5)",
            "centering": "complete_daily_training_cross_section_valid_rules_only",
            "scale": "sqrt(sum(centered_D^2)/valid_training_rule_rows)",
            "clip_in_model": [-5.0, 5.0],
        },
        "source_layout": {
            "sampler": "<sampler-dir>/train.pkl",
            "features": "<feature-root>/train/rule_components.npy",
        },
        **result,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return payload
