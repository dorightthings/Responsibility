"""Prediction normalization and the frozen daily IC-family metrics."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def labels_from_sampler(sampler: Any) -> pd.Series:
    """Read each sample's current-row label without materializing TS windows."""

    idx_map = np.asarray(sampler.idx_map)
    rows = idx_map[:, 0].astype(np.int64, copy=False)
    columns = idx_map[:, 1].astype(np.int64, copy=False)
    positions = np.asarray(sampler.idx_arr[rows, columns])
    if np.isnan(positions).any():
        raise ValueError("current sampler positions contain NaN")
    exact = positions.astype(np.int64)
    if not np.array_equal(positions, exact):
        raise ValueError("current sampler positions are not exact integers")
    labels = np.asarray(sampler.data_arr[exact, -1])
    if len(labels) != len(sampler.get_index()):
        raise ValueError("label count differs from sampler index")
    return pd.Series(labels, index=sampler.get_index(), name="label")


def normalize_prediction(value: Any) -> pd.Series:
    if isinstance(value, pd.DataFrame) and value.shape[1] == 1:
        value = value.iloc[:, 0]
    if not isinstance(value, pd.Series):
        raise TypeError("prediction must be a Series or one-column DataFrame")
    result = value.copy()
    if not isinstance(result.index, pd.MultiIndex):
        raise TypeError("prediction index must be a MultiIndex")
    if list(result.index.names) != ["datetime", "instrument"]:
        raise ValueError("unexpected prediction index names: %s" % result.index.names)
    if result.index.has_duplicates:
        raise ValueError("prediction index contains duplicates")
    values = result.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("prediction contains NaN or Inf")
    result.name = "score"
    return result.sort_index()


def _aggregate(values: Sequence[float]) -> Tuple[float, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("aggregate requires at least one finite value")
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=0))
    ratio = float(mean / std) if len(array) >= 2 and std > 0 else None
    return mean, ratio


def daily_metrics(
    predictions: pd.Series, labels: pd.Series
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compute daily-equal MSE, IC, ICIR, RankIC and RankICIR.

    NaN labels are ignored per metric date.  Population standard deviation is
    used across dates, matching the frozen experiment aggregation.
    """

    predictions = normalize_prediction(predictions)
    labels = labels.sort_index()
    if not predictions.index.equals(labels.index):
        raise ValueError("prediction and label indices differ")
    if np.isinf(labels.to_numpy(dtype=np.float64, copy=False)).any():
        raise FloatingPointError("labels contain Inf")
    frame = pd.DataFrame({"prediction": predictions, "label": labels})
    rows: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    mse_values: List[float] = []
    ic_values: List[float] = []
    rank_values: List[float] = []
    for date, group in frame.groupby(level="datetime", sort=False):
        date_string = str(pd.Timestamp(date).date())
        valid = group.loc[group["label"].notna(), ["prediction", "label"]]
        row: Dict[str, Any] = {
            "datetime": date_string,
            "samples_total": int(len(group)),
            "valid_labels": int(len(valid)),
            "mse": None,
            "ic": None,
            "rankic": None,
        }
        if len(valid) < 2:
            exclusions.append(
                {"datetime": date_string, "metric": "all", "reason": "fewer_than_two_valid_labels"}
            )
            rows.append(row)
            continue
        label_values = valid["label"].to_numpy(dtype=np.float64)
        prediction_values = valid["prediction"].to_numpy(dtype=np.float64)
        label_std = float(np.std(label_values, ddof=1))
        if not math.isfinite(label_std) or label_std <= 0:
            exclusions.append(
                {"datetime": date_string, "metric": "all", "reason": "constant_or_invalid_label"}
            )
            rows.append(row)
            continue
        normalized_label = (label_values - float(np.mean(label_values))) / label_std
        mse = float(np.mean(np.square(prediction_values - normalized_label)))
        if not math.isfinite(mse):
            raise FloatingPointError("non-finite daily MSE on %s" % date_string)
        row["mse"] = mse
        mse_values.append(mse)
        prediction_std = float(np.std(prediction_values, ddof=1))
        if not math.isfinite(prediction_std) or prediction_std <= 0:
            exclusions.append(
                {"datetime": date_string, "metric": "ic/rankic", "reason": "constant_or_invalid_prediction"}
            )
            rows.append(row)
            continue
        ic = float(pd.Series(prediction_values).corr(pd.Series(label_values)))
        rankic = float(
            pd.Series(prediction_values).corr(
                pd.Series(label_values), method="spearman"
            )
        )
        if math.isfinite(ic):
            row["ic"] = ic
            ic_values.append(ic)
        else:
            exclusions.append(
                {"datetime": date_string, "metric": "ic", "reason": "nonfinite_correlation"}
            )
        if math.isfinite(rankic):
            row["rankic"] = rankic
            rank_values.append(rankic)
        else:
            exclusions.append(
                {"datetime": date_string, "metric": "rankic", "reason": "nonfinite_correlation"}
            )
        rows.append(row)

    mse, _ = _aggregate(mse_values)
    ic, icir = _aggregate(ic_values)
    rankic, rankicir = _aggregate(rank_values)
    return (
        {
            "MSE": mse,
            "IC": ic,
            "ICIR": icir,
            "RankIC": rankic,
            "RankICIR": rankicir,
            "dates_total": len(rows),
            "ic_days_used": len(ic_values),
            "rankic_days_used": len(rank_values),
            "aggregation": "within-day correlation then equal-weight date mean; population std for IR",
        },
        rows,
        exclusions,
    )
