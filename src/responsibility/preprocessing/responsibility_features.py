"""Build row-aligned responsibility features from samplers and raw Qlib data.

For each ``train``/``valid``/``test`` sampler this module writes:

* ``stock_state.npy``: ``float32[N, 14]``.  Its first twelve columns are
  standardized using training rows only; the last two validity fields remain
  unscaled.
* ``rule_components.npy``: ``float32[N, 8]`` containing raw common and
  idiosyncratic return components for horizons 1, 5, 20 and 60.

The computation is label-free.  Every output segment is bound to the exact
sampler row order by a deterministic MultiIndex SHA-256.
"""

from __future__ import annotations

import gc
import hashlib
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SCHEMA_VERSION = "heterogeneous-rule-features-v1"
SEGMENTS = ("train", "valid", "test")
DATASET_BENCHMARKS = {
    "short_csi300": "SH000300",
    "short_csi800_direct": "SH000906",
}
CONTINUOUS_COLUMNS = (
    "c_1",
    "c_5",
    "c_20",
    "c_60",
    "e_1",
    "e_5",
    "e_20",
    "e_60",
    "beta",
    "R2",
    "residual_volatility",
    "abnormal_volume_20",
)
STOCK_STATE_COLUMNS = CONTINUOUS_COLUMNS + ("valid_ratio_60", "volume_valid")
RULE_COMPONENT_COLUMNS = (
    "c_1",
    "c_5",
    "c_20",
    "c_60",
    "e_1",
    "e_5",
    "e_20",
    "e_60",
)

ROLLING_WINDOW = 60
MIN_OLS_PAIRS = 40
VARIANCE_FLOOR = 1e-12
AVOL_WINDOW = 20
MIN_AVOL_HISTORY = 10
EPSILON = 1e-12
STD_FLOOR = 1e-6
HISTORY_BUFFER = 119
DEFAULT_CHUNK_ROWS = 262_144


class ResponsibilityFeatureError(RuntimeError):
    """The responsibility-feature data contract could not be satisfied."""


@dataclass(frozen=True)
class SamplerIndex:
    """Compact representation of one sampler's exact public row order."""

    datetime_ns: np.ndarray
    instrument_codes: np.ndarray
    instrument_names: Tuple[str, ...]
    index_sha256: str
    date_min: str
    date_max: str

    def __len__(self) -> int:
        return int(self.datetime_ns.shape[0])


@dataclass(frozen=True)
class ResponsibilityFeatureSegment:
    """Read-only mmap view of one generated feature segment."""

    stock_state: np.ndarray
    rule_components: np.ndarray
    segment: str
    manifest: Mapping[str, Any]

    def __len__(self) -> int:
        return int(self.stock_state.shape[0])

    def __getitem__(self, indexer: Any) -> Tuple[np.ndarray, np.ndarray]:
        return self.stock_state[indexer], self.rule_components[indexer]

    def take(self, positions: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        positions_array = np.asarray(positions, dtype=np.int64)
        return self.stock_state[positions_array], self.rule_components[positions_array]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def multiindex_sha256(index: pd.MultiIndex) -> str:
    """Hash dates and instruments exactly as used by the frozen experiments."""

    digest = hashlib.sha256()
    for date, instrument in index:
        digest.update(str(pd.Timestamp(date).date()).encode("ascii"))
        digest.update(b"\t")
        digest.update(str(instrument).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _normalise_instrument(value: Any) -> str:
    instrument = str(value).strip().upper()
    if not instrument:
        raise ResponsibilityFeatureError("empty instrument identifier")
    return instrument


def _validate_sampler_index(index: Any, label: str) -> pd.MultiIndex:
    if not isinstance(index, pd.MultiIndex):
        raise ResponsibilityFeatureError("%s sampler index is not a MultiIndex" % label)
    if list(index.names) != ["datetime", "instrument"]:
        raise ResponsibilityFeatureError("%s sampler index names differ" % label)
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ResponsibilityFeatureError(
            "%s sampler index must be unique and monotonic" % label
        )
    if len(index) == 0:
        raise ResponsibilityFeatureError("%s sampler is empty" % label)
    return index


def read_sampler_index(path: Path, label: Optional[str] = None) -> pd.MultiIndex:
    """Load and validate a trusted Qlib sampler's public row index."""

    sampler_path = Path(path).expanduser().resolve(strict=True)
    if not sampler_path.is_file() or sampler_path.is_symlink():
        raise ResponsibilityFeatureError("invalid sampler path: %s" % sampler_path)
    with sampler_path.open("rb") as handle:
        sampler = pickle.load(handle)
    if not hasattr(sampler, "get_index"):
        raise ResponsibilityFeatureError("sampler lacks get_index(): %s" % sampler_path)
    index = _validate_sampler_index(
        sampler.get_index(), label or sampler_path.stem
    ).copy()
    del sampler
    gc.collect()
    return index


def _sampler_paths(
    sampler_dir: Path, expected_dataset: Optional[str] = None
) -> Dict[str, Path]:
    root = Path(sampler_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    metadata_path = root / "build_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ResponsibilityFeatureError("sampler build metadata must be an object")
        identity = metadata.get("dataset")
        if expected_dataset is not None and identity != expected_dataset:
            raise ResponsibilityFeatureError(
                "sampler dataset mismatch: expected %s, got %s"
                % (expected_dataset, identity)
            )
        if metadata.get("status") not in (None, "DONE"):
            raise ResponsibilityFeatureError("sampler build metadata is not DONE")
    paths = {segment: root / (segment + ".pkl") for segment in SEGMENTS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing sampler files: " + ", ".join(missing))
    return paths


def _compact_index(path: Path, segment: str) -> SamplerIndex:
    index = read_sampler_index(path, segment)
    digest = multiindex_sha256(index)
    datetime_ns = pd.DatetimeIndex(index.get_level_values("datetime")).asi8.copy()
    instrument_level = index.names.index("instrument")
    instrument_codes = np.asarray(index.codes[instrument_level], dtype=np.int32).copy()
    if np.any(instrument_codes < 0):
        raise ResponsibilityFeatureError(
            "%s sampler contains missing instrument codes" % segment
        )
    instrument_names = tuple(
        _normalise_instrument(value) for value in index.levels[instrument_level]
    )
    if len(set(instrument_names)) != len(instrument_names):
        raise ResponsibilityFeatureError("instrument names collide after upper-casing")
    dates = pd.DatetimeIndex(datetime_ns)
    result = SamplerIndex(
        datetime_ns=datetime_ns,
        instrument_codes=instrument_codes,
        instrument_names=instrument_names,
        index_sha256=digest,
        date_min=str(dates.min().date()),
        date_max=str(dates.max().date()),
    )
    del index
    gc.collect()
    return result


def _position_groups(index: SamplerIndex) -> Dict[str, np.ndarray]:
    codes = index.instrument_codes
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    boundaries = np.flatnonzero(np.diff(sorted_codes)) + 1
    result: Dict[str, np.ndarray] = {}
    for positions in np.split(order, boundaries):
        if positions.size == 0:
            continue
        code = int(codes[int(positions[0])])
        result[index.instrument_names[code]] = positions.astype(np.int64, copy=False)
    return result


def _trailing_sum(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    prefix = np.empty(values.size + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(values, dtype=np.float64, out=prefix[1:])
    ends = np.arange(1, values.size + 1, dtype=np.int64)
    starts = np.maximum(ends - int(window), 0)
    return prefix[ends] - prefix[starts]


def _exclusive_history_sum(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    prefix = np.empty(values.size + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(values, dtype=np.float64, out=prefix[1:])
    positions = np.arange(values.size, dtype=np.int64)
    starts = np.maximum(positions - int(window), 0)
    return prefix[positions] - prefix[starts]


def _log_returns(close: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    close = np.asarray(close, dtype=np.float64)
    returns = np.full(close.shape, np.nan, dtype=np.float64)
    valid = np.zeros(close.shape, dtype=bool)
    pair_valid = (
        np.isfinite(close[1:])
        & (close[1:] > 0.0)
        & np.isfinite(close[:-1])
        & (close[:-1] > 0.0)
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        candidate = np.log(close[1:] / close[:-1])
    pair_valid &= np.isfinite(candidate)
    valid[1:] = pair_valid
    returns[1:][pair_valid] = candidate[pair_valid]
    return returns, valid


def compute_daily_features(
    stock_close: np.ndarray,
    stock_volume: np.ndarray,
    market_returns: np.ndarray,
    market_valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the frozen 14 states and 8 rule components for one stock."""

    stock_close = np.asarray(stock_close, dtype=np.float64)
    stock_volume = np.asarray(stock_volume, dtype=np.float64)
    market_returns = np.asarray(market_returns, dtype=np.float64)
    market_valid = np.asarray(market_valid, dtype=bool)
    size = stock_close.size
    if not (
        stock_volume.shape == (size,)
        and market_returns.shape == (size,)
        and market_valid.shape == (size,)
    ):
        raise ValueError("daily input arrays must have identical one-dimensional shapes")

    stock_returns, stock_price_valid = _log_returns(stock_close)
    volume_valid = np.isfinite(stock_volume) & (stock_volume > 0.0)
    return_pair_valid = stock_price_valid & market_valid
    return_pair_valid[1:] &= volume_valid[1:] & volume_valid[:-1]
    return_pair_valid[0] = False

    pair_float = return_pair_valid.astype(np.float64)
    x = np.where(return_pair_valid, market_returns, 0.0)
    y = np.where(return_pair_valid, stock_returns, 0.0)
    n = _trailing_sum(pair_float, ROLLING_WINDOW)
    sum_x = _trailing_sum(x, ROLLING_WINDOW)
    sum_y = _trailing_sum(y, ROLLING_WINDOW)
    sum_x2 = _trailing_sum(x * x, ROLLING_WINDOW)
    sum_y2 = _trailing_sum(y * y, ROLLING_WINDOW)
    sum_xy = _trailing_sum(x * y, ROLLING_WINDOW)

    alpha = np.full(size, np.nan, dtype=np.float64)
    beta = np.full(size, np.nan, dtype=np.float64)
    r2 = np.full(size, np.nan, dtype=np.float64)
    residual_volatility = np.full(size, np.nan, dtype=np.float64)
    s_mm = np.full(size, np.nan, dtype=np.float64)
    sst = np.full(size, np.nan, dtype=np.float64)
    sse = np.full(size, np.nan, dtype=np.float64)

    enough = n >= float(MIN_OLS_PAIRS)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        s_mm[enough] = sum_x2[enough] - (sum_x[enough] * sum_x[enough]) / n[enough]
        sst[enough] = sum_y2[enough] - (sum_y[enough] * sum_y[enough]) / n[enough]
    candidate = enough & np.isfinite(s_mm) & (
        (s_mm / np.maximum(n, 1.0)) >= VARIANCE_FLOOR
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        covariance_sum = sum_xy - (sum_x * sum_y) / np.maximum(n, 1.0)
        beta[candidate] = covariance_sum[candidate] / s_mm[candidate]
        alpha[candidate] = (
            sum_y[candidate] / n[candidate]
            - beta[candidate] * sum_x[candidate] / n[candidate]
        )
        sse[candidate] = (
            sum_y2[candidate]
            + n[candidate] * alpha[candidate] * alpha[candidate]
            + beta[candidate] * beta[candidate] * sum_x2[candidate]
            - 2.0 * alpha[candidate] * sum_y[candidate]
            - 2.0 * beta[candidate] * sum_xy[candidate]
            + 2.0 * alpha[candidate] * beta[candidate] * sum_x[candidate]
        )

    finite_candidate = (
        candidate
        & np.isfinite(alpha)
        & np.isfinite(beta)
        & np.isfinite(sst)
        & np.isfinite(sse)
    )
    low_stock_variance = finite_candidate & (
        (sst / np.maximum(n, 1.0)) < VARIANCE_FLOOR
    )
    regular_stock_variance = finite_candidate & ~low_stock_variance
    r2[low_stock_variance] = 0.0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        r2[regular_stock_variance] = np.clip(
            1.0 - sse[regular_stock_variance] / sst[regular_stock_variance], 0.0, 1.0
        )
        residual_volatility[finite_candidate] = np.sqrt(
            np.maximum(sse[finite_candidate], 0.0) / (n[finite_candidate] - 2.0)
        )
    ols_valid = finite_candidate & np.isfinite(r2) & np.isfinite(residual_volatility)
    alpha[~ols_valid] = np.nan
    beta[~ols_valid] = np.nan
    r2[~ols_valid] = np.nan
    residual_volatility[~ols_valid] = np.nan

    decomposition_valid = ols_valid & return_pair_valid
    common = np.zeros(size, dtype=np.float64)
    residual = np.zeros(size, dtype=np.float64)
    common[decomposition_valid] = beta[decomposition_valid] * market_returns[decomposition_valid]
    residual[decomposition_valid] = (
        stock_returns[decomposition_valid]
        - alpha[decomposition_valid]
        - common[decomposition_valid]
    )
    decomposition_valid &= np.isfinite(common) & np.isfinite(residual)
    common[~decomposition_valid] = 0.0
    residual[~decomposition_valid] = 0.0

    horizons = (1, 5, 20, 60)
    common_sums = np.column_stack([_trailing_sum(common, h) for h in horizons])
    residual_sums = np.column_stack([_trailing_sum(residual, h) for h in horizons])
    rule_components = np.column_stack((common_sums, residual_sums))
    decomposition_valid &= np.all(np.isfinite(rule_components), axis=1)
    rule_components[~decomposition_valid, :] = 0.0

    history_count = _exclusive_history_sum(volume_valid.astype(np.float64), AVOL_WINDOW)
    history_volume_sum = _exclusive_history_sum(
        np.where(volume_valid, stock_volume, 0.0), AVOL_WINDOW
    )
    avol_valid = volume_valid & (history_count >= float(MIN_AVOL_HISTORY))
    avol = np.full(size, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        history_mean = history_volume_sum / np.maximum(history_count, 1.0)
        avol[avol_valid] = np.log(
            (stock_volume[avol_valid] + EPSILON)
            / (history_mean[avol_valid] + EPSILON)
        )
    avol_valid &= np.isfinite(avol)
    avol[~avol_valid] = np.nan

    raw_state = np.full((size, 14), np.nan, dtype=np.float64)
    raw_state[:, :8] = rule_components
    raw_state[~decomposition_valid, :8] = np.nan
    raw_state[:, 8] = beta
    raw_state[:, 9] = r2
    raw_state[:, 10] = residual_volatility
    raw_state[:, 11] = avol
    raw_state[:, 12] = n / float(ROLLING_WINDOW)
    raw_state[:, 13] = avol_valid.astype(np.float64)
    return raw_state, rule_components


def _training_statistics(raw_state_path: Path, chunk_rows: int) -> Dict[str, np.ndarray]:
    raw = np.load(str(raw_state_path), mmap_mode="r", allow_pickle=False)
    if raw.ndim != 2 or raw.shape[1] != 14 or raw.dtype != np.dtype("float64"):
        raise ResponsibilityFeatureError("invalid temporary train state array")
    counts = np.zeros(12, dtype=np.int64)
    sums = np.zeros(12, dtype=np.float64)
    for start in range(0, raw.shape[0], chunk_rows):
        block = np.asarray(raw[start : start + chunk_rows, :12], dtype=np.float64)
        finite = np.isfinite(block)
        counts += finite.sum(axis=0, dtype=np.int64)
        sums += np.where(finite, block, 0.0).sum(axis=0, dtype=np.float64)
    if np.any(counts == 0):
        missing = [CONTINUOUS_COLUMNS[i] for i in np.flatnonzero(counts == 0)]
        raise ResponsibilityFeatureError(
            "train has no valid values for fields: %s" % missing
        )
    means = sums / counts.astype(np.float64)

    squared_deviation_sums = np.zeros(12, dtype=np.float64)
    for start in range(0, raw.shape[0], chunk_rows):
        block = np.asarray(raw[start : start + chunk_rows, :12], dtype=np.float64)
        finite = np.isfinite(block)
        deviations = np.where(finite, block - means, 0.0)
        squared_deviation_sums += (deviations * deviations).sum(axis=0, dtype=np.float64)
    standard_deviations = np.sqrt(
        squared_deviation_sums / counts.astype(np.float64)
    )
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(standard_deviations)):
        raise ResponsibilityFeatureError("non-finite train standardization statistic")
    scales = np.maximum(standard_deviations, STD_FLOOR)
    return {
        "count": counts,
        "mean": means,
        "std": standard_deviations,
        "scale": scales,
    }


def _write_standardized_state(
    raw_state_path: Path,
    output_path: Path,
    statistics: Mapping[str, np.ndarray],
    chunk_rows: int,
) -> None:
    raw = np.load(str(raw_state_path), mmap_mode="r", allow_pickle=False)
    output = np.lib.format.open_memmap(
        str(output_path), mode="w+", dtype=np.float32, shape=raw.shape
    )
    means = np.asarray(statistics["mean"], dtype=np.float64)
    scales = np.asarray(statistics["scale"], dtype=np.float64)
    for start in range(0, raw.shape[0], chunk_rows):
        stop = min(start + chunk_rows, raw.shape[0])
        source = np.asarray(raw[start:stop], dtype=np.float64)
        target = np.zeros(source.shape, dtype=np.float32)
        for column in range(12):
            valid = np.isfinite(source[:, column])
            normalized = (source[valid, column] - means[column]) / scales[column]
            cast = normalized.astype(np.float32)
            if not np.all(np.isfinite(cast)):
                raise ResponsibilityFeatureError("float32 overflow in standardized state")
            target[valid, column] = cast
        flags = source[:, 12:14]
        if not np.all(np.isfinite(flags)):
            raise ResponsibilityFeatureError("non-finite validity field")
        target[:, 12:14] = flags.astype(np.float32)
        output[start:stop] = target
    output.flush()
    del output, raw


def _raw_series(
    raw_quotes: pd.DataFrame, instrument: str, calendar: pd.DatetimeIndex
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        frame = raw_quotes.xs(instrument, level="instrument")
    except KeyError:
        return (
            np.full(len(calendar), np.nan, dtype=np.float64),
            np.full(len(calendar), np.nan, dtype=np.float64),
        )
    if frame.index.has_duplicates:
        raise ResponsibilityFeatureError(
            "raw provider returned duplicate dates for %s" % instrument
        )
    frame = frame.reindex(calendar)
    return (
        frame["$close"].to_numpy(dtype=np.float64, copy=True),
        frame["$volume"].to_numpy(dtype=np.float64, copy=True),
    )


def _query_raw_quotes(
    provider: Path,
    instruments: Iterable[str],
    first_sample_date: pd.Timestamp,
    last_sample_date: pd.Timestamp,
) -> Tuple[pd.DatetimeIndex, pd.DataFrame]:
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    qlib.init(provider_uri=str(provider), region=REG_CN)
    full_calendar = pd.DatetimeIndex(
        D.calendar(end_time=last_sample_date, freq="day")
    )
    sample_position = int(full_calendar.get_indexer([first_sample_date])[0])
    last_position = int(full_calendar.get_indexer([last_sample_date])[0])
    if sample_position < HISTORY_BUFFER or last_position < sample_position:
        raise ResponsibilityFeatureError(
            "provider calendar cannot supply the required 119-day history buffer"
        )
    calendar = full_calendar[sample_position - HISTORY_BUFFER : last_position + 1]
    if len(calendar) == 0 or calendar[HISTORY_BUFFER] != first_sample_date:
        raise ResponsibilityFeatureError("incorrect history-buffer alignment")

    requested = sorted({_normalise_instrument(value) for value in instruments})
    raw = D.features(
        requested,
        ["$close", "$volume"],
        start_time=calendar[0],
        end_time=calendar[-1],
        freq="day",
    )
    if not isinstance(raw, pd.DataFrame) or list(raw.index.names) != [
        "instrument",
        "datetime",
    ]:
        raise ResponsibilityFeatureError("unexpected D.features output")
    if list(raw.columns) != ["$close", "$volume"]:
        raw = raw.reindex(columns=["$close", "$volume"])
    instrument_level = raw.index.names.index("instrument")
    normalized_levels = pd.Index(
        [_normalise_instrument(value) for value in raw.index.levels[instrument_level]]
    )
    if normalized_levels.has_duplicates:
        raise ResponsibilityFeatureError(
            "raw instruments collide after upper-casing"
        )
    original_levels = pd.Index(raw.index.levels[instrument_level].astype(str))
    if not original_levels.equals(normalized_levels):
        raw.index = raw.index.set_levels(
            normalized_levels, level=instrument_level, verify_integrity=True
        )
    if raw.index.has_duplicates:
        raise ResponsibilityFeatureError(
            "raw provider returned duplicate instrument/date rows"
        )
    if not raw.index.is_monotonic_increasing:
        raw = raw.sort_index()
    return calendar, raw


def build_responsibility_feature_store(
    dataset: str,
    sampler_dir: Path,
    qlib_provider: Path,
    output_root: Path,
    benchmark: Optional[str] = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> Dict[str, Any]:
    """Build a complete feature store in a new output directory."""

    if dataset not in DATASET_BENCHMARKS:
        raise ValueError("unsupported dataset: %s" % dataset)
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    expected_benchmark = DATASET_BENCHMARKS[dataset]
    if benchmark is not None and _normalise_instrument(benchmark) != expected_benchmark:
        raise ValueError(
            "benchmark for %s must be %s" % (dataset, expected_benchmark)
        )
    benchmark_code = expected_benchmark
    sampler_paths = _sampler_paths(sampler_dir, expected_dataset=dataset)
    provider = Path(qlib_provider).expanduser().resolve(strict=True)
    if not provider.is_dir():
        raise NotADirectoryError(provider)
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    indexes: Dict[str, SamplerIndex] = {}
    groups: Dict[str, Dict[str, np.ndarray]] = {}
    for segment in SEGMENTS:
        indexes[segment] = _compact_index(sampler_paths[segment], segment)
        groups[segment] = _position_groups(indexes[segment])

    first_sample_date = min(pd.Timestamp(index.date_min) for index in indexes.values())
    last_sample_date = max(pd.Timestamp(index.date_max) for index in indexes.values())
    stock_instruments = sorted(
        set().union(*(set(segment_groups) for segment_groups in groups.values()))
    )
    calendar, raw_quotes = _query_raw_quotes(
        provider,
        stock_instruments + [benchmark_code],
        first_sample_date,
        last_sample_date,
    )
    calendar_position: Dict[str, np.ndarray] = {}
    for segment, index in indexes.items():
        positions = calendar.get_indexer(pd.DatetimeIndex(index.datetime_ns))
        if np.any(positions < 0):
            raise ResponsibilityFeatureError(
                "%s sampler date is absent from provider calendar" % segment
            )
        calendar_position[segment] = positions.astype(np.int64, copy=False)

    market_close, _ = _raw_series(raw_quotes, benchmark_code, calendar)
    market_returns, market_valid = _log_returns(market_close)
    if int(market_valid.sum()) < ROLLING_WINDOW:
        raise ResponsibilityFeatureError(
            "benchmark has insufficient valid raw close history"
        )

    raw_state_paths: Dict[str, Path] = {}
    raw_state_maps: Dict[str, np.memmap] = {}
    component_maps: Dict[str, np.memmap] = {}
    for segment in SEGMENTS:
        segment_dir = output_root / segment
        segment_dir.mkdir(exist_ok=False)
        raw_state_path = segment_dir / ".raw_state.tmp.npy"
        raw_state_paths[segment] = raw_state_path
        raw_state_maps[segment] = np.lib.format.open_memmap(
            str(raw_state_path),
            mode="w+",
            dtype=np.float64,
            shape=(len(indexes[segment]), 14),
        )
        raw_state_maps[segment][:] = np.nan
        component_maps[segment] = np.lib.format.open_memmap(
            str(segment_dir / "rule_components.npy"),
            mode="w+",
            dtype=np.float32,
            shape=(len(indexes[segment]), 8),
        )
        component_maps[segment][:] = 0.0

    for instrument in stock_instruments:
        stock_close, stock_volume = _raw_series(raw_quotes, instrument, calendar)
        daily_state, daily_components = compute_daily_features(
            stock_close, stock_volume, market_returns, market_valid
        )
        for segment in SEGMENTS:
            sample_positions = groups[segment].get(instrument)
            if sample_positions is None:
                continue
            daily_positions = calendar_position[segment][sample_positions]
            raw_state_maps[segment][sample_positions] = daily_state[daily_positions]
            component_values = daily_components[daily_positions].astype(np.float32)
            if not np.all(np.isfinite(component_values)):
                raise ResponsibilityFeatureError("float32 overflow in rule components")
            component_maps[segment][sample_positions] = component_values

    for segment in SEGMENTS:
        raw_state_maps[segment].flush()
        component_maps[segment].flush()
    del raw_state_maps, component_maps, raw_quotes
    gc.collect()

    for segment in SEGMENTS:
        raw_state = np.load(
            str(raw_state_paths[segment]), mmap_mode="r", allow_pickle=False
        )
        if not np.all(np.isfinite(raw_state[:, 12:14])):
            raise ResponsibilityFeatureError("unfilled sampler rows in %s" % segment)
        del raw_state

    statistics = _training_statistics(raw_state_paths["train"], chunk_rows)
    for segment in SEGMENTS:
        _write_standardized_state(
            raw_state_paths[segment],
            output_root / segment / "stock_state.npy",
            statistics,
            chunk_rows,
        )
        raw_state_paths[segment].unlink()

    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "dataset_id": dataset,
        "boundary": "no_purge",
        "benchmark": benchmark_code,
        "source_layout": {
            "samplers": "<sampler-dir>/{train,valid,test}.pkl",
            "qlib_provider": "external CLI path",
        },
        "builder_sha256": file_sha256(Path(__file__).resolve()),
        "raw_fields": ["$close", "$volume"],
        "calendar": {
            "first": str(calendar[0].date()),
            "first_sample": str(first_sample_date.date()),
            "last_sample": str(last_sample_date.date()),
            "history_buffer_trading_days": HISTORY_BUFFER,
        },
        "stock_state_columns": list(STOCK_STATE_COLUMNS),
        "rule_component_columns": list(RULE_COMPONENT_COLUMNS),
        "standardization": {
            "source_segment": "train",
            "before_label_tail_deletion": True,
            "ddof": 0,
            "std_floor": STD_FLOOR,
            "count": statistics["count"].astype(int).tolist(),
            "mean": statistics["mean"].tolist(),
            "std": statistics["std"].tolist(),
            "scale": statistics["scale"].tolist(),
        },
        "segments": {},
    }
    for segment in SEGMENTS:
        manifest["segments"][segment] = {
            "rows": len(indexes[segment]),
            "index_sha256": indexes[segment].index_sha256,
            "date_min": indexes[segment].date_min,
            "date_max": indexes[segment].date_max,
            "sampler_filename": sampler_paths[segment].name,
            "stock_state": {
                "path": "%s/stock_state.npy" % segment,
                "shape": [len(indexes[segment]), 14],
                "dtype": "float32",
            },
            "rule_components": {
                "path": "%s/rule_components.npy" % segment,
                "shape": [len(indexes[segment]), 8],
                "dtype": "float32",
            },
        }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_feature_segment(
    root: Path,
    segment: str,
    expected_index: Optional[pd.MultiIndex] = None,
) -> ResponsibilityFeatureSegment:
    """Load one generated segment as read-only memory-mapped arrays."""

    feature_root = Path(root).expanduser().resolve(strict=True)
    manifest_path = feature_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ResponsibilityFeatureError("unsupported feature-store schema")
    if segment not in SEGMENTS or segment not in manifest.get("segments", {}):
        raise ResponsibilityFeatureError("unknown segment: %s" % segment)
    record = manifest["segments"][segment]
    stock_state = np.load(
        str(feature_root / record["stock_state"]["path"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    rule_components = np.load(
        str(feature_root / record["rule_components"]["path"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    expected_rows = int(record["rows"])
    if stock_state.shape != (expected_rows, 14) or stock_state.dtype != np.dtype(
        "float32"
    ):
        raise ResponsibilityFeatureError("stock_state array violates manifest")
    if rule_components.shape != (expected_rows, 8) or rule_components.dtype != np.dtype(
        "float32"
    ):
        raise ResponsibilityFeatureError("rule_components array violates manifest")
    if expected_index is not None:
        _validate_sampler_index(expected_index, "expected")
        if len(expected_index) != expected_rows:
            raise ResponsibilityFeatureError("expected sampler row count differs")
        if multiindex_sha256(expected_index) != record["index_sha256"]:
            raise ResponsibilityFeatureError("expected sampler index SHA differs")
    return ResponsibilityFeatureSegment(
        stock_state=stock_state,
        rule_components=rule_components,
        segment=segment,
        manifest=manifest,
    )
