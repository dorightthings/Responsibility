"""Qlib Top30/Drop30 evaluation for a frozen Responsibility signal."""

from __future__ import annotations

import inspect
import math
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .io import atomic_json, atomic_text, utc_now
from .metrics import normalize_prediction


def _risk_value(frame: pd.DataFrame, name: str) -> float:
    return float(frame.loc[name, "risk"])


def run_backtest(
    prediction_path: Path,
    *,
    provider: Path,
    config: ExperimentConfig,
    output_dir: Path,
    allow_other_qlib_version: bool = False,
) -> Dict[str, Any]:
    """Evaluate one bridge+test signal and write a new backtest directory."""

    prediction_path = Path(prediction_path).expanduser().resolve(strict=True)
    provider = Path(provider).expanduser().resolve(strict=True)
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite backtest: %s" % output_dir)
    if config.topk != 30 or config.n_drop != 30:
        raise ValueError("released evaluation requires Top30/Drop30")
    if config.deal_price != "close":
        raise ValueError("released evaluation requires close deal price")

    signal = normalize_prediction(pd.read_pickle(prediction_path))
    dates = pd.DatetimeIndex(
        signal.index.get_level_values("datetime").unique()
    ).sort_values()
    if len(dates) < 2 or dates[0] != pd.Timestamp(config.bridge_date):
        raise ValueError("signal must begin with the configured bridge date")
    if dates[1] != pd.Timestamp(config.backtest_start_date):
        raise ValueError(
            "first test date differs: %s != %s"
            % (dates[1].date(), config.backtest_start_date)
        )

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    import qlib
    from qlib.backtest import backtest, get_exchange
    from qlib.config import C
    from qlib.contrib.evaluate import risk_analysis

    if str(qlib.__version__) != "0.9.7" and not allow_other_qlib_version:
        raise RuntimeError(
            "validated Qlib version is 0.9.7, found %s; pass "
            "--allow-other-qlib-version only if metric drift is acceptable"
            % qlib.__version__
        )
    qlib.init(provider_uri=str(provider), region="cn")
    exchange_parameters = inspect.signature(get_exchange).parameters
    exchange_defaults = {
        "open_cost": float(exchange_parameters["open_cost"].default),
        "close_cost": float(exchange_parameters["close_cost"].default),
        "min_cost": float(exchange_parameters["min_cost"].default),
        "limit_threshold": float(C.limit_threshold),
        "trade_unit": int(C.trade_unit),
    }
    strategy = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {
            "signal": signal.to_frame("score"),
            "topk": 30,
            "n_drop": 30,
        },
    }
    executor = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    }
    portfolio, indicators = backtest(
        start_time=config.backtest_start_date,
        end_time=config.backtest_end_date,
        strategy=strategy,
        executor=executor,
        benchmark=config.benchmark,
        account=config.account,
        exchange_kwargs={"deal_price": "close"},
    )
    report, positions = portfolio["1day"]
    required = ["return", "bench", "turnover", "cost"]
    if report.empty or not set(required).issubset(report.columns):
        raise RuntimeError("Qlib returned an empty or incomplete daily report")
    if not np.isfinite(report[required].to_numpy(dtype=np.float64)).all():
        raise FloatingPointError("Qlib report contains NaN or Inf")
    excess_no_cost = report["return"] - report["bench"]
    excess_with_cost = excess_no_cost - report["cost"]
    risk_no_cost = risk_analysis(excess_no_cost, freq="1day")
    risk_with_cost = risk_analysis(excess_with_cost, freq="1day")
    metrics = {
        "AR": _risk_value(risk_no_cost, "annualized_return"),
        "IR": _risk_value(risk_no_cost, "information_ratio"),
        "AR_with_cost": _risk_value(risk_with_cost, "annualized_return"),
        "IR_with_cost": _risk_value(risk_with_cost, "information_ratio"),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("Qlib produced non-finite AR/IR")

    output_dir.mkdir(parents=True)
    report.to_pickle(output_dir / "report_normal_1day.pkl")
    pd.to_pickle(positions, output_dir / "positions_normal_1day.pkl")
    pd.concat(
        {
            "excess_return_without_cost": risk_no_cost,
            "excess_return_with_cost": risk_with_cost,
        }
    ).to_pickle(output_dir / "port_analysis_1day.pkl")
    if "1day" in indicators:
        pd.to_pickle(
            indicators["1day"][0], output_dir / "indicators_normal_1day.pkl"
        )
        pd.to_pickle(
            indicators["1day"][1],
            output_dir / "indicators_normal_1day_obj.pkl",
        )
    result = {
        "status": "DONE",
        "finished_utc": utc_now(),
        "prediction": str(prediction_path),
        "provider": str(provider),
        "dataset": config.dataset,
        "qlib_version": str(qlib.__version__),
        "benchmark": config.benchmark,
        "start_date": config.backtest_start_date,
        "end_date": config.backtest_end_date,
        "report_dates": int(len(report)),
        "strategy": "TopkDropoutStrategy",
        "topk": 30,
        "n_drop": 30,
        "deal_price": "close",
        "headline": "excess_return_without_cost",
        "exchange_defaults": exchange_defaults,
        "metrics": metrics,
    }
    atomic_json(output_dir / "result.json", result)
    atomic_text(output_dir / "DONE", result["finished_utc"] + "\n")
    return result
