"""Portable loaders for Qlib samplers and responsibility sidecars."""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch


SEGMENTS = ("train", "valid", "test")


@dataclass(frozen=True)
class DataPaths:
    data_root: Path
    manifest: Path
    samplers: Mapping[str, Path]
    responsibility_root: Path
    direction_scale: Path
    provider: Path


@dataclass(frozen=True)
class ResponsibilitySegment:
    stock_state: np.ndarray
    rule_components: np.ndarray
    record: Mapping[str, Any]


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object: %s" % path)
    return value


def _dataset_record(manifest: Mapping[str, Any], dataset: str) -> Mapping[str, Any]:
    datasets = manifest.get("datasets")
    if isinstance(datasets, dict):
        if dataset not in datasets:
            raise ValueError("data manifest does not declare dataset %s" % dataset)
        record = datasets[dataset]
        if record is None:
            return {}
        if not isinstance(record, dict):
            raise ValueError("data manifest dataset record must be an object")
        return record
    if isinstance(datasets, list):
        for value in datasets:
            if value == dataset:
                return {}
            if isinstance(value, dict) and value.get("id") == dataset:
                return value
    raise ValueError("data manifest does not declare dataset %s" % dataset)


def _relative(root: Path, value: Any, default: str) -> Path:
    if isinstance(value, Mapping):
        value = value.get("path")
    raw = Path(str(value if value is not None else default)).expanduser()
    return (root / raw).resolve() if not raw.is_absolute() else raw.resolve()


def resolve_data_paths(data_root: Path, dataset: str) -> DataPaths:
    """Resolve all large local assets through ``data/manifest.json``.

    The manifest may omit individual paths when the documented standard
    layout is used.  Large arrays and samplers therefore remain outside Git
    while the same CLI works with a copied directory or local symlinks.
    """

    root = Path(data_root).expanduser().resolve(strict=True)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "missing data manifest: %s; copy data/manifest.example.json and "
            "fill in local assets" % manifest_path
        )
    manifest = read_json(manifest_path)
    record = _dataset_record(manifest, dataset)
    sampler_record = record.get("samplers", {})
    if not isinstance(sampler_record, dict):
        raise ValueError("samplers record must be an object")
    samplers = {
        segment: _relative(
            root,
            sampler_record.get(segment),
            "samplers/%s/%s.pkl" % (dataset, segment),
        )
        for segment in SEGMENTS
    }
    responsibility_root = _relative(
        root,
        record.get("responsibility_root", record.get("responsibility")),
        "responsibility/%s" % dataset,
    )
    direction_scale = _relative(
        root,
        record.get("direction_scale"),
        "direction_scales/%s.json" % dataset,
    )
    provider_record = manifest.get("qlib_provider", record.get("qlib_provider"))
    provider = _relative(root, provider_record, "qlib/cn_data")
    required = list(samplers.values()) + [
        responsibility_root / "manifest.json",
        direction_scale,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing data assets: " + ", ".join(missing))
    return DataPaths(
        data_root=root,
        manifest=manifest_path,
        samplers=samplers,
        responsibility_root=responsibility_root,
        direction_scale=direction_scale,
        provider=provider,
    )


def load_sampler(path: Path) -> Any:
    with Path(path).open("rb") as handle:
        sampler = pickle.load(handle)
    required = (
        "data_arr",
        "idx_arr",
        "idx_map",
        "nan_idx",
        "step_len",
        "fillna_type",
        "get_index",
    )
    missing = [name for name in required if not hasattr(sampler, name)]
    if missing:
        raise TypeError("incompatible Qlib sampler %s; missing %s" % (path, missing))
    validate_index(sampler.get_index(), str(path))
    return sampler


def validate_index(index: Any, label: str) -> pd.MultiIndex:
    if not isinstance(index, pd.MultiIndex):
        raise TypeError("%s index must be a pandas MultiIndex" % label)
    if list(index.names) != ["datetime", "instrument"]:
        raise ValueError("%s has unexpected index names %s" % (label, index.names))
    if index.has_duplicates:
        raise ValueError("%s index contains duplicates" % label)
    dates = pd.DatetimeIndex(index.get_level_values("datetime"))
    if not dates.is_monotonic_increasing:
        raise ValueError("%s must be grouped chronologically by datetime" % label)
    return index


def multiindex_sha256(index: pd.MultiIndex) -> str:
    digest = hashlib.sha256()
    for date, instrument in index:
        digest.update(str(pd.Timestamp(date).date()).encode("ascii"))
        digest.update(b"\t")
        digest.update(str(instrument).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_responsibility_segment(
    root: Path, segment: str, expected_index: pd.MultiIndex
) -> ResponsibilitySegment:
    if segment not in SEGMENTS:
        raise ValueError("unknown segment: %s" % segment)
    manifest = read_json(Path(root) / "manifest.json")
    segments = manifest.get("segments", {})
    if segment not in segments or not isinstance(segments[segment], dict):
        raise ValueError("responsibility manifest lacks segment %s" % segment)
    record = segments[segment]
    state_record = record.get("stock_state", {})
    rule_record = record.get("rule_components", {})
    state_path = Path(root) / str(
        state_record.get("path", "%s/stock_state.npy" % segment)
    )
    rule_path = Path(root) / str(
        rule_record.get("path", "%s/rule_components.npy" % segment)
    )
    stock_state = np.load(str(state_path), mmap_mode="r", allow_pickle=False)
    rule_components = np.load(str(rule_path), mmap_mode="r", allow_pickle=False)
    rows = len(expected_index)
    if stock_state.shape != (rows, 14) or stock_state.dtype != np.float32:
        raise ValueError(
            "stock_state must be float32[%d,14], got %s %s"
            % (rows, stock_state.shape, stock_state.dtype)
        )
    if rule_components.shape != (rows, 8) or rule_components.dtype != np.float32:
        raise ValueError(
            "rule_components must be float32[%d,8], got %s %s"
            % (rows, rule_components.shape, rule_components.dtype)
        )
    expected_hash = record.get("index_sha256")
    if expected_hash and multiindex_sha256(expected_index) != str(expected_hash):
        raise ValueError("responsibility sidecar index differs for %s" % segment)
    return ResponsibilitySegment(stock_state, rule_components, record)


def load_direction_scale(
    path: Path, dataset: str, device: torch.device
) -> Tuple[torch.Tensor, Mapping[str, Any]]:
    payload = read_json(path)
    identity = payload.get("dataset", payload.get("dataset_id"))
    if identity is not None and str(identity) != dataset:
        raise ValueError("direction scale dataset mismatch")
    source = payload.get("source_segment")
    if source is not None and str(source) != "train":
        raise ValueError("direction scale must be derived from the train segment")
    values = np.asarray(payload.get("direction_scale"), dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("direction_scale must contain four finite values")
    if np.any(values <= 1e-12):
        raise ValueError("direction_scale entries must be greater than 1e-12")
    return torch.as_tensor(values, dtype=torch.float32, device=device), payload


def _ffill_2d(values: np.ndarray) -> np.ndarray:
    mask = np.isnan(values.astype(float, copy=False))
    fill_index = np.where(
        ~mask, np.arange(values.shape[1], dtype=np.int64)[None, :], 0
    )
    np.maximum.accumulate(fill_index, axis=1, out=fill_index)
    return np.take_along_axis(values, fill_index, axis=1)


def _build_window_index(sampler: Any, chunk_size: int = 200_000) -> np.ndarray:
    sample_count = len(sampler.idx_map)
    step_len = int(sampler.step_len)
    if step_len <= 0:
        raise ValueError("sampler step_len must be positive")
    result = np.empty((sample_count, step_len), dtype=np.int32)
    offsets = np.arange(-step_len + 1, 1, dtype=np.int64)
    for begin in range(0, sample_count, chunk_size):
        end = min(begin + chunk_size, sample_count)
        row_col = sampler.idx_map[begin:end]
        rows = row_col[:, 0].astype(np.int64, copy=False)[:, None] + offsets
        cols = row_col[:, 1].astype(np.int64, copy=False)[:, None]
        padded = rows < 0
        indices = sampler.idx_arr[np.maximum(rows, 0), cols]
        if padded.any():
            indices[padded] = np.nan
        if sampler.fillna_type in ("ffill", "ffill+bfill"):
            indices = _ffill_2d(indices)
            if sampler.fillna_type == "ffill+bfill":
                indices = _ffill_2d(indices[:, ::-1])[:, ::-1]
        elif sampler.fillna_type != "none":
            raise ValueError("unsupported sampler fillna_type: %s" % sampler.fillna_type)
        result[begin:end] = np.nan_to_num(
            indices.astype(np.float64, copy=False), nan=sampler.nan_idx
        ).astype(np.int32)
    return result


class CachedSampler:
    """Exact TSDataSampler window lookup cached on CPU or one CUDA device."""

    def __init__(self, sampler: Any, device: torch.device, cache_on_gpu: bool):
        self.sampler = sampler
        self.device = device
        started = time.perf_counter()
        window_index = _build_window_index(sampler)
        self.cache_on_gpu = bool(cache_on_gpu and device.type == "cuda")
        if self.cache_on_gpu:
            source = np.asarray(sampler.data_arr)
            if source.dtype != np.float32:
                raise TypeError("CUDA cache requires float32 sampler data")
            self.data_arr = torch.from_numpy(source).to(device)
            self.window_index = torch.from_numpy(
                window_index.astype(np.int64, copy=False)
            ).to(device)
            torch.cuda.synchronize(device)
        else:
            self.data_arr = np.asarray(sampler.data_arr)
            self.window_index = window_index
        self.build_seconds = time.perf_counter() - started

    def get(self, positions: np.ndarray) -> torch.Tensor:
        positions = np.asarray(positions, dtype=np.int64)
        if positions.ndim != 1 or positions.size == 0:
            raise ValueError("daily positions must be a non-empty vector")
        if self.cache_on_gpu:
            begin, end = int(positions[0]), int(positions[-1]) + 1
            if end - begin == positions.size:
                lookup = self.window_index[begin:end]
            else:
                lookup = self.window_index.index_select(
                    0, torch.as_tensor(positions, dtype=torch.long, device=self.device)
                )
            return self.data_arr[lookup]
        value = self.data_arr[self.window_index[positions]]
        return torch.as_tensor(value, device=self.device)


def daily_plan(index: pd.MultiIndex) -> Tuple[np.ndarray, np.ndarray]:
    dates = pd.DatetimeIndex(index.get_level_values("datetime"))
    values = dates.asi8
    change = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate(([0], change)).astype(np.int64)
    ends = np.concatenate((change, [len(values)])).astype(np.int64)
    counts = ends - starts
    if int(counts.sum()) != len(index) or np.any(counts <= 0):
        raise ValueError("could not form contiguous daily batches")
    return starts, counts


class DailyFeatureLoader:
    """Yield one complete stock cross-section and its aligned sidecars."""

    def __init__(
        self,
        sampler: Any,
        sidecar: ResponsibilitySegment,
        device: torch.device,
        *,
        shuffle: bool,
        cache_on_gpu: bool,
    ) -> None:
        self.index = validate_index(sampler.get_index(), "sampler")
        self.starts, self.counts = daily_plan(self.index)
        self.cached = CachedSampler(sampler, device, cache_on_gpu)
        self.sidecar = sidecar
        self.device = device
        self.shuffle = bool(shuffle)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]]:
        order = np.arange(len(self.starts))
        if self.shuffle:
            np.random.shuffle(order)
        for item in order:
            start = int(self.starts[item])
            positions = np.arange(start, start + int(self.counts[item]))
            data = self.cached.get(positions)
            state = torch.as_tensor(
                np.asarray(self.sidecar.stock_state[positions]),
                dtype=torch.float32,
                device=self.device,
            )
            rules = torch.as_tensor(
                np.asarray(self.sidecar.rule_components[positions]),
                dtype=torch.float32,
                device=self.device,
            )
            yield data, state, rules, positions

    def __len__(self) -> int:
        return len(self.starts)


def sampler_description(sampler: Any) -> Dict[str, Any]:
    index = validate_index(sampler.get_index(), "sampler")
    dates = pd.DatetimeIndex(index.get_level_values("datetime"))
    return {
        "rows": int(len(index)),
        "dates": int(dates.nunique()),
        "date_min": str(dates.min().date()),
        "date_max": str(dates.max().date()),
        "index_sha256": multiindex_sha256(index),
    }
