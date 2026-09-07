"""Configuration validation for the grouped-learning-rate Responsibility route."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping


DATASET_DEFAULTS: Mapping[str, Mapping[str, Any]] = {
    "short_csi300": {
        "beta": 5.0,
        "benchmark": "SH000300",
        "bridge_date": "2020-06-30",
        "backtest_start_date": "2020-07-01",
        "backtest_end_date": "2022-12-31",
    },
    "short_csi800_direct": {
        "beta": 2.0,
        "benchmark": "SH000906",
        "bridge_date": "2020-06-30",
        "backtest_start_date": "2020-07-01",
        "backtest_end_date": "2022-12-31",
    },
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated configuration for the grouped-LR development starting point.

    Scientific defaults are deliberately strict.  A changed beta, stopping
    threshold, epoch cap, or Top30/Drop30 policy is a new experiment, not a
    reproduction of the released route.
    """

    source_path: str
    dataset: str
    beta: float
    benchmark: str
    bridge_date: str
    backtest_start_date: str
    backtest_end_date: str
    d_feat: int = 158
    d_model: int = 256
    t_nhead: int = 4
    s_nhead: int = 2
    temporal_dropout: float = 0.5
    cross_section_dropout: float = 0.5
    gate_input_start_index: int = 158
    gate_input_end_index: int = 221
    gate_mode: str = "shared"
    adapter_rank: int = 32
    alpha_initial: float = 0.1
    rho_initial: float = 0.1
    max_epochs: int = 150
    smoke_epochs: int = 2
    threshold: float = 0.95
    base_learning_rate: float = 1e-5
    crfr_learning_rate: float = 1e-4
    rrca_adapter_learning_rate: float = 1e-5
    rrca_condition_learning_rate: float = 1e-4
    gradient_clip_value: float = 3.0
    selection_temperature_start: float = 1.0
    selection_temperature_end: float = 0.5
    selection_temperature_anneal_epochs: int = 10
    topk: int = 30
    n_drop: int = 30
    deal_price: str = "close"
    account: float = 100_000_000.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _read_mapping(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError(
                "PyYAML is required for YAML configs; install the released "
                "environment or use a JSON config"
            ) from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("config root must be a mapping: %s" % path)
    return value


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("config section %s must be a mapping" % name)
    return value


def _dataset_id(payload: Mapping[str, Any]) -> str:
    value = payload.get("dataset")
    if isinstance(value, dict):
        value = value.get("id", value.get("name"))
    if value is None and isinstance(payload.get("data"), dict):
        value = payload["data"].get("dataset")
    dataset = str(value or "").strip()
    if dataset not in DATASET_DEFAULTS:
        raise ValueError(
            "dataset must be one of %s, got %r"
            % (sorted(DATASET_DEFAULTS), dataset)
        )
    return dataset


def _pick(
    payload: Mapping[str, Any], section: Mapping[str, Any], key: str, default: Any
) -> Any:
    return payload[key] if key in payload else section.get(key, default)


def _close(actual: float, expected: float, label: str) -> None:
    if not math.isfinite(float(actual)) or abs(float(actual) - float(expected)) > 1e-15:
        raise ValueError(
            "%s=%r changes the frozen protocol; expected %r"
            % (label, actual, expected)
        )


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load YAML/JSON and reject silent changes to the published protocol."""

    resolved = Path(path).expanduser().resolve(strict=True)
    payload = _read_mapping(resolved)
    dataset = _dataset_id(payload)
    defaults = DATASET_DEFAULTS[dataset]
    model = _section(payload, "model")
    training = _section(payload, "training")
    backtest = _section(payload, "backtest")
    if "learning_rate" in payload or "learning_rate" in training:
        raise ValueError(
            "training.learning_rate belongs to the uniform-LR release; "
            "use the current grouped-LR configs instead of silently mixing versions"
        )

    config = ExperimentConfig(
        source_path=str(resolved),
        dataset=dataset,
        beta=float(_pick(payload, model, "beta", defaults["beta"])),
        benchmark=str(
            _pick(payload, backtest, "benchmark", defaults["benchmark"])
        ),
        bridge_date=str(
            _pick(payload, backtest, "bridge_date", defaults["bridge_date"])
        ),
        backtest_start_date=str(
            backtest.get("start_date", defaults["backtest_start_date"])
        ),
        backtest_end_date=str(
            backtest.get("end_date", defaults["backtest_end_date"])
        ),
        d_feat=int(model.get("d_feat", 158)),
        d_model=int(model.get("d_model", 256)),
        t_nhead=int(model.get("t_nhead", 4)),
        s_nhead=int(model.get("s_nhead", 2)),
        temporal_dropout=float(
            model.get("temporal_dropout", model.get("T_dropout_rate", 0.5))
        ),
        cross_section_dropout=float(
            model.get(
                "cross_section_dropout", model.get("S_dropout_rate", 0.5)
            )
        ),
        gate_input_start_index=int(model.get("gate_input_start_index", 158)),
        gate_input_end_index=int(model.get("gate_input_end_index", 221)),
        gate_mode=str(model.get("gate_mode", "shared")),
        adapter_rank=int(model.get("adapter_rank", 32)),
        alpha_initial=float(model.get("alpha_initial", 0.1)),
        rho_initial=float(model.get("rho_initial", 0.1)),
        max_epochs=int(training.get("max_epochs", 150)),
        smoke_epochs=int(training.get("smoke_epochs", 2)),
        threshold=float(training.get("threshold", 0.95)),
        base_learning_rate=float(training.get("base_learning_rate", 1e-5)),
        crfr_learning_rate=float(training.get("crfr_learning_rate", 1e-4)),
        rrca_adapter_learning_rate=float(
            training.get("rrca_adapter_learning_rate", 1e-5)
        ),
        rrca_condition_learning_rate=float(
            training.get("rrca_condition_learning_rate", 1e-4)
        ),
        gradient_clip_value=float(training.get("gradient_clip_value", 3.0)),
        selection_temperature_start=float(
            training.get("selection_temperature_start", 1.0)
        ),
        selection_temperature_end=float(
            training.get("selection_temperature_end", 0.5)
        ),
        selection_temperature_anneal_epochs=int(
            training.get("selection_temperature_anneal_epochs", 10)
        ),
        topk=int(backtest.get("topk", 30)),
        n_drop=int(backtest.get("n_drop", 30)),
        deal_price=str(backtest.get("deal_price", "close")),
        account=float(backtest.get("account", 100_000_000)),
    )

    _close(config.beta, float(defaults["beta"]), "beta")
    _close(config.threshold, 0.95, "threshold")
    _close(config.base_learning_rate, 1e-5, "base_learning_rate")
    _close(config.crfr_learning_rate, 1e-4, "crfr_learning_rate")
    _close(config.rrca_adapter_learning_rate, 1e-5, "rrca_adapter_learning_rate")
    _close(
        config.rrca_condition_learning_rate,
        1e-4,
        "rrca_condition_learning_rate",
    )
    expected_model = {
        "d_feat": 158,
        "d_model": 256,
        "t_nhead": 4,
        "s_nhead": 2,
        "temporal_dropout": 0.5,
        "cross_section_dropout": 0.5,
        "gate_input_start_index": 158,
        "gate_input_end_index": 221,
        "adapter_rank": 32,
    }
    for key, expected in expected_model.items():
        if getattr(config, key) != expected:
            raise ValueError(
                "%s=%r changes the released architecture; expected %r"
                % (key, getattr(config, key), expected)
            )
    if config.gate_mode != "shared":
        raise ValueError("the retained CRFR route requires gate_mode='shared'")
    if config.max_epochs != 150 or config.smoke_epochs != 2:
        raise ValueError("the frozen epoch caps are formal=150 and smoke=2")
    if config.topk != 30 or config.n_drop != 30:
        raise ValueError("the released backtest contract is Top30/Drop30")
    if config.benchmark != defaults["benchmark"]:
        raise ValueError(
            "benchmark mismatch for %s: %s" % (dataset, config.benchmark)
        )
    if config.deal_price != "close":
        raise ValueError("the released backtest uses deal_price='close'")
    if config.selection_temperature_anneal_epochs < 2:
        raise ValueError("selection temperature anneal must span at least 2 epochs")
    return config
