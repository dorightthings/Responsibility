"""Build the frozen Alpha158 + Market63 Qlib samplers.

The builder deliberately accepts the Qlib provider and output directory as
arguments.  It contains no machine-specific data path and never overwrites an
existing dataset directory or pickle.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import logging
import os
import pickle
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

LABEL = "Ref($close, -5) / Ref($close, -1) - 1"
STEP_LEN = 8
STOCK_DIM = 158
MARKET_DIM = 63
LABEL_DIM = 1
TOTAL_DIM = STOCK_DIM + MARKET_DIM + LABEL_DIM

SEGMENTS: Mapping[str, Tuple[str, str]] = {
    "train": ("2008-01-01", "2020-03-31"),
    "valid": ("2020-04-01", "2020-06-30"),
    "test": ("2020-07-01", "2022-12-31"),
}
DATA_KEYS: Mapping[str, str] = {
    "train": "learn",
    "valid": "infer",
    "test": "infer",
}


@dataclass(frozen=True)
class StageExpectation:
    samples: int
    dates: int
    date_min: str
    date_max: str
    label_nan_samples: int


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    instruments: str
    benchmark: str
    universe_semantics: str
    stages: Mapping[str, StageExpectation]


DATASET_SPECS: Mapping[str, DatasetSpec] = {
    "short_csi300": DatasetSpec(
        dataset_id="short_csi300",
        instruments="csi300",
        benchmark="SH000300",
        universe_semantics="point-in-time CSI300 membership from the supplied provider",
        stages={
            "train": StageExpectation(856246, 2979, "2008-01-02", "2020-03-31", 0),
            "valid": StageExpectation(17700, 59, "2020-04-01", "2020-06-30", 53),
            "test": StageExpectation(183300, 611, "2020-07-01", "2022-12-30", 305),
        },
    ),
    "short_csi800_direct": DatasetSpec(
        dataset_id="short_csi800_direct",
        instruments="csi800",
        benchmark="SH000906",
        universe_semantics=(
            "author-compatible direct provider csi800; known upstream historical "
            "truncation is intentionally preserved; the frozen protocol does not "
            "synthesize CSI300+CSI500"
        ),
        stages={
            "train": StageExpectation(1757732, 2979, "2008-01-02", "2020-03-31", 0),
            "valid": StageExpectation(47200, 59, "2020-04-01", "2020-06-30", 161),
            "test": StageExpectation(488721, 611, "2020-07-01", "2022-12-30", 1198),
        },
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def market_feature_config() -> Tuple[List[str], List[str]]:
    """Return the frozen Market63 expression and name order."""

    fields: List[str] = []
    names: List[str] = []
    for code in ("sh000300", "sh000905", "sh000906"):
        fields.append('Mask($close/Ref($close,1)-1, "%s")' % code)
        names.append("%s_ret1" % code)
        for window in (5, 10, 20, 30, 60):
            fields.extend(
                [
                    'Mask(Mean($close/Ref($close,1)-1,%d), "%s")'
                    % (window, code),
                    'Mask(Std($close/Ref($close,1)-1,%d), "%s")'
                    % (window, code),
                    'Mask(Mean($amount,%d)/$amount, "%s")' % (window, code),
                    'Mask(Std($amount,%d)/$amount, "%s")' % (window, code),
                ]
            )
            names.extend(
                [
                    "%s_ret_mean_%d" % (code, window),
                    "%s_ret_std_%d" % (code, window),
                    "%s_amount_mean_ratio_%d" % (code, window),
                    "%s_amount_std_ratio_%d" % (code, window),
                ]
            )
    if len(fields) != MARKET_DIM or len(names) != MARKET_DIM:
        raise AssertionError("Market feature configuration must be 63-dimensional")
    return fields, names


def infer_processor_config() -> List[Dict[str, Any]]:
    """Return fresh copies of the frozen feature preprocessing configuration."""

    return [
        {
            "class": "RobustZScoreNorm",
            "kwargs": {"fields_group": "feature", "clip_outlier": True},
        },
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ]


def learn_processor_config() -> List[Dict[str, Any]]:
    return [{"class": "DropnaLabel"}]


def _normalise_dataset_ids(dataset_ids: Sequence[str]) -> Tuple[str, ...]:
    selected = tuple(str(value) for value in dataset_ids)
    if not selected:
        raise ValueError("at least one dataset must be selected")
    if len(set(selected)) != len(selected):
        raise ValueError("dataset selections must not contain duplicates")
    unknown = [value for value in selected if value not in DATASET_SPECS]
    if unknown:
        raise ValueError("unknown datasets: %s" % ", ".join(unknown))
    return selected


def _provider_required_files(
    provider: Path, dataset_ids: Sequence[str]
) -> Mapping[str, Path]:
    required: Dict[str, Path] = {"calendars/day.txt": provider / "calendars/day.txt"}
    for dataset_id in dataset_ids:
        instruments = DATASET_SPECS[dataset_id].instruments
        relative = "instruments/%s.txt" % instruments
        required[relative] = provider / relative
    for code in ("sh000300", "sh000905", "sh000906"):
        for field in ("close", "amount"):
            relative = "features/%s/%s.day.bin" % (code, field)
            required[relative] = provider / relative
    return required


def validate_provider(provider: Path, dataset_ids: Sequence[str]) -> Path:
    """Validate the minimum Qlib files required by the frozen build."""

    selected = _normalise_dataset_ids(dataset_ids)
    root = Path(provider).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    missing = [
        relative
        for relative, path in _provider_required_files(root, selected).items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Qlib provider lacks required files: %s" % ", ".join(missing)
        )
    return root


def reorder_sampler_by_datetime(sampler: Any) -> Any:
    """Make each cross-sectional day a contiguous sampler block."""

    index = sampler.get_index()
    if not isinstance(index, pd.MultiIndex):
        raise TypeError("sampler index must be a pandas MultiIndex")
    if list(index.names) != ["datetime", "instrument"]:
        raise ValueError("unexpected sampler index names: %s" % list(index.names))
    if len(index) == 0:
        raise ValueError("sampler is empty")
    dates = pd.DatetimeIndex(index.get_level_values("datetime"))
    instruments = index.get_level_values("instrument").astype(str).to_numpy()
    order = np.lexsort((instruments, dates.asi8))
    sampler.idx_map = sampler.idx_map[order]
    sampler.data_index = sampler.data_index[order]

    sorted_index = sampler.get_index()
    if not sorted_index.is_monotonic_increasing:
        raise AssertionError("sampler index is not datetime/instrument sorted")
    sorted_dates = pd.DatetimeIndex(sorted_index.get_level_values("datetime"))
    transitions = int((sorted_dates[1:] != sorted_dates[:-1]).sum()) + 1
    if transitions != sorted_dates.nunique():
        raise AssertionError("sampler dates are not contiguous")
    return sampler


def make_responsibility_dataset_class() -> Any:
    """Create the Qlib dataset class lazily so CLI help has no Qlib side effects."""

    from qlib.contrib.data.handler import check_transform_proc
    from qlib.data.dataset import DatasetH, TSDataSampler, TSDatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    class MarketContextDataHandler(DataHandlerLP):
        def __init__(
            self,
            instruments: Any = "csi300",
            start_time: Any = None,
            end_time: Any = None,
            freq: str = "day",
            infer_processors: Any = None,
            learn_processors: Any = None,
            fit_start_time: Any = None,
            fit_end_time: Any = None,
            process_type: Any = DataHandlerLP.PTYPE_A,
            filter_pipe: Any = None,
            inst_processors: Any = None,
            **kwargs: Any,
        ) -> None:
            checked_infer = check_transform_proc(
                infer_processors or [], fit_start_time, fit_end_time
            )
            checked_learn = check_transform_proc(
                learn_processors or [], fit_start_time, fit_end_time
            )
            data_loader = {
                "class": "QlibDataLoader",
                "kwargs": {
                    "config": {"feature": market_feature_config()},
                    "filter_pipe": filter_pipe,
                    "freq": freq,
                    "inst_processors": inst_processors,
                },
            }
            super().__init__(
                instruments=instruments,
                start_time=start_time,
                end_time=end_time,
                data_loader=data_loader,
                infer_processors=checked_infer,
                learn_processors=checked_learn,
                process_type=process_type,
                **kwargs,
            )

    class ResponsibilityTSDatasetH(TSDatasetH):
        def __init__(
            self,
            market_data_handler_config: Any = None,
            sequence_fillna_type: str = "ffill+bfill",
            **kwargs: Any,
        ) -> None:
            if sequence_fillna_type not in {"ffill", "ffill+bfill", "none"}:
                raise ValueError("unsupported sequence fill: %s" % sequence_fillna_type)
            self.sequence_fillna_type = sequence_fillna_type
            super().__init__(**kwargs)
            market_handler = MarketContextDataHandler(
                **(market_data_handler_config or {})
            )
            self.market_dataset = DatasetH(market_handler, segments=self.segments)

        def _prepare_seg(self, slc: slice, **kwargs: Any) -> Any:
            dtype = kwargs.pop("dtype", None)
            if not isinstance(slc, slice):
                slc = slice(*slc)
            flt_col = kwargs.pop("flt_col", None)
            if flt_col is None:
                flt_col = self.flt_col

            ext_slice = self._extend_slice(slc, self.cal, self.step_len)
            data = DatasetH._prepare_seg(self, ext_slice, **kwargs)
            market = self.market_dataset.prepare(
                ext_slice, col_set="feature", data_key=DataHandlerLP.DK_I
            )
            if isinstance(market.columns, pd.MultiIndex):
                market_names = list(market.columns.get_level_values(-1))
            else:
                market_names = [str(column) for column in market.columns]
            if market.shape[1] != MARKET_DIM:
                raise AssertionError(
                    "expected %d market columns, got %d"
                    % (MARKET_DIM, market.shape[1])
                )
            market = pd.DataFrame(
                market.values,
                index=market.index,
                columns=pd.MultiIndex.from_tuples(
                    [("feature", name) for name in market_names]
                ),
            ).reindex(data.index)

            level_zero = data.columns.get_level_values(0)
            stock_features = data.loc[:, level_zero == "feature"]
            labels = data.loc[:, level_zero == "label"]
            if stock_features.shape[1] != STOCK_DIM or labels.shape[1] != LABEL_DIM:
                raise AssertionError(
                    "expected Alpha158 + one label, got %d + %d columns"
                    % (stock_features.shape[1], labels.shape[1])
                )
            combined = pd.concat([stock_features, market, labels], axis=1)

            filter_kwargs = copy.deepcopy(kwargs)
            if flt_col is not None:
                filter_kwargs["col_set"] = flt_col
                filter_data = DatasetH._prepare_seg(self, ext_slice, **filter_kwargs)
                if len(filter_data.columns) != 1:
                    raise AssertionError("filter data must contain exactly one column")
            else:
                filter_data = None

            sampler = TSDataSampler(
                data=combined,
                start=slc.start,
                end=slc.stop,
                step_len=self.step_len,
                fillna_type=self.sequence_fillna_type,
                dtype=dtype,
                flt_data=filter_data,
            )
            return reorder_sampler_by_datetime(sampler)

    ResponsibilityTSDatasetH.__name__ = "ResponsibilityTSDatasetH"
    return ResponsibilityTSDatasetH


def _build_dataset(spec: DatasetSpec) -> Any:
    from qlib.contrib.data.handler import Alpha158

    infer_processors = infer_processor_config()
    handler = Alpha158(
        instruments=spec.instruments,
        start_time=SEGMENTS["train"][0],
        end_time=SEGMENTS["test"][1],
        fit_start_time=SEGMENTS["train"][0],
        fit_end_time=SEGMENTS["train"][1],
        infer_processors=infer_processors,
        learn_processors=learn_processor_config(),
        label=[LABEL],
    )
    market_config = {
        "start_time": SEGMENTS["train"][0],
        "end_time": SEGMENTS["test"][1],
        "fit_start_time": SEGMENTS["train"][0],
        "fit_end_time": SEGMENTS["train"][1],
        "instruments": spec.instruments,
        "infer_processors": infer_processor_config(),
        "learn_processors": [],
    }
    dataset_class = make_responsibility_dataset_class()
    return dataset_class(
        handler=handler,
        segments=dict(SEGMENTS),
        step_len=STEP_LEN,
        sequence_fillna_type="ffill+bfill",
        market_data_handler_config=market_config,
    )


def _current_label_nan_count(sampler: Any) -> int:
    row_column = np.asarray(sampler.idx_map)
    if row_column.ndim != 2 or row_column.shape[1] != 2:
        raise AssertionError("sampler idx_map must have shape [samples, 2]")
    data_indices = sampler.idx_arr[row_column[:, 0], row_column[:, 1]]
    if np.isnan(data_indices).any():
        raise AssertionError("current sample rows contain missing data indices")
    labels = sampler.data_arr[data_indices.astype(np.int64), -1]
    return int(np.isnan(labels).sum())


def audit_sampler(sampler: Any, spec: DatasetSpec, stage: str) -> Dict[str, Any]:
    """Enforce the frozen public index, shape, and stage-statistics contract."""

    index = sampler.get_index()
    if not isinstance(index, pd.MultiIndex):
        raise TypeError("sampler index must be a pandas MultiIndex")
    if list(index.names) != ["datetime", "instrument"]:
        raise AssertionError("unexpected index names: %s" % list(index.names))
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise AssertionError("sampler index must be unique and chronologically sorted")
    if len(index) == 0:
        raise AssertionError("sampler is empty")
    if int(sampler.step_len) != STEP_LEN or sampler.fillna_type != "ffill+bfill":
        raise AssertionError("sampler sequence contract changed")
    if sampler.data_arr.ndim != 2 or sampler.data_arr.shape[1] != TOTAL_DIM:
        raise AssertionError("sampler data must have %d columns" % TOTAL_DIM)
    if sampler.data_arr.dtype != np.float32:
        raise AssertionError("sampler data dtype must be float32")
    first = sampler[0]
    if first.shape != (STEP_LEN, TOTAL_DIM) or first.dtype != np.float32:
        raise AssertionError("sample shape/dtype contract changed")

    dates = pd.DatetimeIndex(index.get_level_values("datetime"))
    transitions = int((dates[1:] != dates[:-1]).sum()) + 1
    summary: Dict[str, Any] = {
        "data_key": DATA_KEYS[stage],
        "samples": int(len(index)),
        "dates": int(dates.nunique()),
        "date_min": str(dates.min().date()),
        "date_max": str(dates.max().date()),
        "instruments": int(index.get_level_values("instrument").nunique()),
        "label_nan_samples": _current_label_nan_count(sampler),
        "datetime_contiguous": transitions == dates.nunique(),
        "step_len": STEP_LEN,
        "columns": TOTAL_DIM,
        "dtype": "float32",
        "fillna_type": "ffill+bfill",
    }
    expected = spec.stages[stage]
    for field in (
        "samples",
        "dates",
        "date_min",
        "date_max",
        "label_nan_samples",
    ):
        actual_value = summary[field]
        expected_value = getattr(expected, field)
        if actual_value != expected_value:
            raise AssertionError(
                "%s/%s differs from the frozen provider contract: %s=%r, expected %r"
                % (spec.dataset_id, stage, field, actual_value, expected_value)
            )
    if not summary["datetime_contiguous"]:
        raise AssertionError("sampler dates are not contiguous")
    return summary


class _DigestingWriter:
    def __init__(self, handle: Any) -> None:
        self.handle = handle
        self.digest = hashlib.sha256()

    def write(self, value: bytes) -> int:
        self.digest.update(value)
        return self.handle.write(value)


def dump_pickle_new(value: Any, output: Path) -> Dict[str, Any]:
    """Write one pickle once, computing SHA-256 without a second large-file pass."""

    if output.exists():
        raise FileExistsError(output)
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    with partial.open("xb") as raw:
        writer = _DigestingWriter(raw)
        pickle.dump(value, writer, protocol=pickle.HIGHEST_PROTOCOL)
        raw.flush()
        os.fsync(raw.fileno())
        digest = writer.digest.hexdigest()
    partial.rename(output)
    return {
        "relative_path": output.name,
        "bytes": int(output.stat().st_size),
        "sha256": digest,
        "pickle_protocol": pickle.HIGHEST_PROTOCOL,
    }


def _write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_text_new(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _provider_control_hashes(
    provider: Path, dataset_ids: Sequence[str]
) -> Mapping[str, str]:
    return {
        relative: sha256_file(path)
        for relative, path in _provider_required_files(provider, dataset_ids).items()
    }


def _metadata_base(
    spec: DatasetSpec, provider_control_hashes: Mapping[str, str]
) -> Dict[str, Any]:
    import qlib

    market_fields, market_names = market_feature_config()
    return {
        "schema_version": "responsibility-sampler-build-v1",
        "status": "IN_PROGRESS",
        "started_utc": utc_now(),
        "dataset": spec.dataset_id,
        "instruments": spec.instruments,
        "benchmark": spec.benchmark,
        "universe_semantics": spec.universe_semantics,
        "segments": {key: list(value) for key, value in SEGMENTS.items()},
        "data_keys": dict(DATA_KEYS),
        "label": LABEL,
        "step_len": STEP_LEN,
        "dimensions": {
            "stock": STOCK_DIM,
            "market": MARKET_DIM,
            "label": LABEL_DIM,
            "total": TOTAL_DIM,
        },
        "stock_handler": "qlib.contrib.data.handler.Alpha158",
        "market_fields": market_fields,
        "market_names": market_names,
        "infer_processors": infer_processor_config(),
        "learn_processors": learn_processor_config(),
        "sequence_fillna_type": "ffill+bfill",
        "strict_frozen_statistics": True,
        "provider_path_recorded": False,
        "provider_control_sha256": dict(provider_control_hashes),
        "software": {
            "python": platform.python_version(),
            "pyqlib": qlib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "stage_results": {},
    }


def _build_one_dataset(
    spec: DatasetSpec,
    output_root: Path,
    provider_control_hashes: Mapping[str, str],
) -> Mapping[str, Any]:
    from qlib.data.dataset.handler import DataHandlerLP

    dataset_dir = output_root / spec.dataset_id
    dataset_dir.mkdir(parents=False, exist_ok=False)
    metadata = _metadata_base(spec, provider_control_hashes)
    try:
        LOGGER.info("building handlers for %s", spec.dataset_id)
        dataset = _build_dataset(spec)
        qlib_data_keys = {
            "train": DataHandlerLP.DK_L,
            "valid": DataHandlerLP.DK_I,
            "test": DataHandlerLP.DK_I,
        }
        for stage in ("train", "valid", "test"):
            LOGGER.info("preparing %s/%s (%s)", spec.dataset_id, stage, DATA_KEYS[stage])
            sampler = dataset.prepare(
                stage,
                col_set=["feature", "label"],
                data_key=qlib_data_keys[stage],
                dtype=np.float32,
            )
            stage_result = audit_sampler(sampler, spec, stage)
            stage_result["pickle"] = dump_pickle_new(
                sampler, dataset_dir / (stage + ".pkl")
            )
            metadata["stage_results"][stage] = stage_result
            LOGGER.info(
                "finished %s/%s: %d samples",
                spec.dataset_id,
                stage,
                stage_result["samples"],
            )
            del sampler
            gc.collect()
        del dataset
        gc.collect()
        metadata["status"] = "DONE"
        metadata["finished_utc"] = utc_now()
        _write_json_new(dataset_dir / "build_metadata.json", metadata)
        _write_text_new(dataset_dir / "DONE", metadata["finished_utc"] + "\n")
        return {
            "dataset": spec.dataset_id,
            "status": "DONE",
            "output": str(dataset_dir),
            "stage_results": metadata["stage_results"],
        }
    except BaseException as exc:
        failure = {
            "schema_version": "responsibility-sampler-build-v1",
            "status": "FAILED",
            "failed_utc": utc_now(),
            "dataset": spec.dataset_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "partial_files_preserved": True,
        }
        failed_path = dataset_dir / "FAILED.json"
        if not failed_path.exists():
            _write_json_new(failed_path, failure)
        raise


def build_sampler_sets(
    provider: Path,
    output_root: Path,
    dataset_ids: Sequence[str] = tuple(DATASET_SPECS),
) -> Mapping[str, Any]:
    """Build complete train/valid/test sampler sets for selected datasets.

    ``output_root`` is normally ``data/samplers``.  Each selected dataset is
    created with ``mkdir(exist_ok=False)``; failed and completed attempts are
    retained and are never overwritten.
    """

    selected = _normalise_dataset_ids(dataset_ids)
    provider_root = validate_provider(provider, selected)
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    conflicts = [str(destination / value) for value in selected if (destination / value).exists()]
    if conflicts:
        raise FileExistsError("dataset output already exists: %s" % ", ".join(conflicts))

    import qlib

    LOGGER.info("initialising Qlib provider supplied by CLI")
    qlib.init(provider_uri=str(provider_root), region="cn")
    controls = _provider_control_hashes(provider_root, selected)
    results: Dict[str, Any] = {}
    for dataset_id in selected:
        spec = DATASET_SPECS[dataset_id]
        relevant = {
            key: value
            for key, value in controls.items()
            if not key.startswith("instruments/")
            or key == "instruments/%s.txt" % spec.instruments
        }
        results[dataset_id] = _build_one_dataset(spec, destination, relevant)
    return {
        "schema_version": "responsibility-sampler-build-v1",
        "status": "DONE",
        "datasets": results,
    }
