"""One-seed training runner for the retained CRFR + RRCA route."""

from __future__ import annotations

import gc
import math
import os
import platform
import random
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig
from .data import (
    DailyFeatureLoader,
    load_direction_scale,
    load_responsibility_segment,
    load_sampler,
    resolve_data_paths,
    sampler_description,
)
from .io import atomic_csv, atomic_json, atomic_text, utc_now
from .metrics import daily_metrics, labels_from_sampler, normalize_prediction


PROTOCOL = "responsibility-crfr-rrca-uniform-lr-portable-v2"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def resolve_device(gpu: int) -> torch.device:
    if gpu < 0:
        raise ValueError("gpu must be non-negative")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu >= torch.cuda.device_count():
        raise ValueError(
            "gpu index %d is unavailable; visible CUDA devices=%d"
            % (gpu, torch.cuda.device_count())
        )
    torch.cuda.set_device(gpu)
    return torch.device("cuda:%d" % gpu)


def _model(config: ExperimentConfig, seed: int, device: torch.device) -> torch.nn.Module:
    # Delayed import keeps `python scripts/train.py --help` useful even before
    # the editable package has been installed.
    from responsibility import build_model

    value = build_model(
        d_feat=config.d_feat,
        d_model=config.d_model,
        t_nhead=config.t_nhead,
        s_nhead=config.s_nhead,
        T_dropout_rate=config.temporal_dropout,
        S_dropout_rate=config.cross_section_dropout,
        gate_input_start_index=config.gate_input_start_index,
        gate_input_end_index=config.gate_input_end_index,
        beta=config.beta,
        rule_seed=seed,
        selection_temperature=config.selection_temperature_start,
        alpha_initial=config.alpha_initial,
        rho_initial=config.rho_initial,
        adapter_rank=config.adapter_rank,
        gate_mode=config.gate_mode,
    )
    if not isinstance(value, torch.nn.Module):
        raise TypeError("build_model() must return torch.nn.Module")
    return value.to(device)


def _parameter_names(model: torch.nn.Module, method: str) -> Tuple[str, ...]:
    function = getattr(model, method, None)
    if not callable(function):
        raise AttributeError("ResponsibilityModel must expose %s()" % method)
    return tuple(str(name) for name in function())


def build_optimizer(
    model: torch.nn.Module, config: ExperimentConfig
) -> Tuple[torch.optim.Optimizer, Dict[str, Any]]:
    """Use one Adam group; component names below are only parameter counts."""
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite")
    named = list(model.named_parameters())
    all_names = {name for name, _ in named}
    g1_names = set(_parameter_names(model, "g1_module_parameter_names"))
    context_names = set(_parameter_names(model, "context_body_parameter_names"))
    condition_names = set(_parameter_names(model, "condition_parameter_names"))
    groups = (g1_names, context_names, condition_names)
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("CRFR, RRCA adapter and RRCA condition groups overlap")
    unknown = set().union(*groups) - all_names
    if unknown:
        raise ValueError("model parameter groups name unknown tensors: %s" % sorted(unknown))
    master_names = all_names - set().union(*groups)
    if not all((master_names, g1_names, context_names, condition_names)):
        raise ValueError("all four model components must have named parameters")

    def parameters(names: set) -> List[torch.nn.Parameter]:
        return [parameter for name, parameter in named if name in names]

    master_parameters = parameters(master_names)
    context_parameters = parameters(context_names)
    g1_parameters = parameters(g1_names)
    condition_parameters = parameters(condition_names)
    trainable_parameters = [
        parameter for _, parameter in named if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("the model has no trainable parameters")
    optimizer = torch.optim.Adam(trainable_parameters, lr=config.learning_rate)
    counts = {
        "total": int(sum(parameter.numel() for _, parameter in named)),
        "backbone": int(sum(parameter.numel() for parameter in master_parameters)),
        "crfr": int(sum(parameter.numel() for parameter in g1_parameters)),
        "rrca_adapters": int(
            sum(parameter.numel() for parameter in context_parameters)
        ),
        "rrca_condition": int(
            sum(parameter.numel() for parameter in condition_parameters)
        ),
    }
    if sum(counts[key] for key in counts if key != "total") != counts["total"]:
        raise ValueError("parameter groups do not cover the full model")
    return optimizer, counts


def drop_extreme_and_zscore(label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mirror the released daily 2.5% tail deletion and sample z-score."""

    _, order = label.sort()
    each_tail = int(0.025 * label.shape[0])
    if each_tail <= 0:
        raise ValueError("daily cross-section is too small for 2.5% tail deletion")
    mask = torch.zeros_like(label, dtype=torch.bool)
    mask[order[each_tail:-each_tail]] = True
    retained = label[mask]
    normalized = (retained - retained.mean()).div(retained.std())
    if retained.numel() < 2 or not torch.isfinite(normalized).all():
        raise FloatingPointError("invalid labels after daily tail deletion/z-score")
    return mask, normalized


def selection_temperature(config: ExperimentConfig, epoch: int) -> float:
    span = config.selection_temperature_anneal_epochs - 1
    fraction = min(max(epoch - 1, 0) / float(span), 1.0)
    return config.selection_temperature_start + fraction * (
        config.selection_temperature_end - config.selection_temperature_start
    )


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DailyFeatureLoader,
    direction_scale: torch.Tensor,
    temperature: float,
    gradient_clip_value: float,
) -> Dict[str, Any]:
    model.train()
    daily_losses: List[torch.Tensor] = []
    conflicts: List[torch.Tensor] = []
    updates = 0
    for data, state, rules, _ in loader:
        feature = data[:, :, :-1]
        label = data[:, -1, -1]
        mask, normalized = drop_extreme_and_zscore(label)
        prediction, auxiliary = model(
            feature[mask].float(),
            state[mask].float(),
            rules[mask].float(),
            direction_scale,
            return_aux=True,
            fusion_enabled=True,
            selection_temperature=temperature,
        )
        loss = torch.mean(torch.square(prediction - normalized))
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), gradient_clip_value)
        optimizer.step()
        daily_losses.append(loss.detach())
        conflict = auxiliary.get("conflict_loss")
        if conflict is not None:
            conflicts.append(torch.as_tensor(conflict).detach().mean())
        updates += 1
    if not daily_losses:
        raise RuntimeError("training epoch produced no optimizer updates")
    return {
        "TrainMSE": float(torch.stack(daily_losses).double().mean().cpu()),
        "conflict_diagnostic": (
            float(torch.stack(conflicts).double().mean().cpu())
            if conflicts
            else None
        ),
        "optimizer_updates": updates,
        "selection_temperature": float(temperature),
    }


def predict_segment(
    model: torch.nn.Module,
    sampler: Any,
    sidecar: Any,
    device: torch.device,
    direction_scale: torch.Tensor,
    temperature: float,
    cache_on_gpu: bool,
) -> pd.Series:
    loader = DailyFeatureLoader(
        sampler,
        sidecar,
        device,
        shuffle=False,
        cache_on_gpu=cache_on_gpu,
    )
    chunks: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for data, state, rules, _ in loader:
            prediction = model(
                data[:, :, :-1].float(),
                state.float(),
                rules.float(),
                direction_scale,
                fusion_enabled=True,
                selection_temperature=temperature,
            )
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("non-finite prediction")
            chunks.append(prediction.detach().reshape(-1).cpu().numpy())
    if not chunks:
        raise RuntimeError("inference produced no predictions")
    return normalize_prediction(
        pd.Series(
            np.concatenate(chunks), index=sampler.get_index(), name="score"
        )
    )


def _release_memory(*values: Any) -> None:
    for value in values:
        del value
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_one_seed(
    config: ExperimentConfig,
    *,
    data_root: Path,
    seed: int,
    gpu: int,
    mode: str,
    output_dir: Path,
    cache_on_gpu: bool = True,
) -> Dict[str, Any]:
    """Train, predict, and compute IC-family metrics for one seed.

    Portfolio AR/IR is intentionally handled by ``scripts/evaluate.py`` so a
    missing Qlib provider cannot invalidate an otherwise complete training run.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative")
    if mode not in ("smoke", "formal"):
        raise ValueError("mode must be smoke or formal")
    run_dir = Path(output_dir).expanduser().resolve()
    if run_dir.exists():
        raise FileExistsError("refusing to overwrite existing run: %s" % run_dir)
    run_dir.mkdir(parents=True)
    checkpoint_dir = run_dir / "checkpoints"
    prediction_dir = run_dir / "predictions"
    metric_dir = run_dir / "metrics"
    for path in (checkpoint_dir, prediction_dir, metric_dir):
        path.mkdir()

    started_utc = utc_now()
    started = time.perf_counter()
    device = resolve_device(gpu)
    set_global_seed(seed)
    paths = resolve_data_paths(data_root, config.dataset)
    max_epochs = config.smoke_epochs if mode == "smoke" else config.max_epochs
    atomic_json(
        run_dir / "run_config.json",
        {
            "protocol": PROTOCOL,
            "status": "RUNNING",
            "started_utc": started_utc,
            "command": [sys.executable, *sys.argv],
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": str(device),
            "cuda_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "seed": seed,
            "mode": mode,
            "config": config.to_dict(),
            "data_manifest": str(paths.manifest),
            "provider": str(paths.provider),
            "cache_on_gpu": bool(cache_on_gpu and device.type == "cuda"),
            "scientific_scope": "development/repeated-test for the released 2020-2022 windows",
        },
    )

    model = _model(config, seed, device)
    optimizer, parameter_counts = build_optimizer(model, config)
    optimizer_metadata = {
        "name": "Adam",
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "parameter_group_count": len(optimizer.param_groups),
        "trainable_parameter_count": int(
            sum(
                parameter.numel()
                for parameter in optimizer.param_groups[0]["params"]
            )
        ),
        "scope": "all_trainable_model_parameters",
        "scheduler": None,
        "gradient_clip_value": config.gradient_clip_value,
        "conflict_weight": 0.0,
    }
    atomic_json(run_dir / "optimizer.json", optimizer_metadata)
    direction_scale, scale_payload = load_direction_scale(
        paths.direction_scale, config.dataset, device
    )

    train_sampler = load_sampler(paths.samplers["train"])
    train_sidecar = load_responsibility_segment(
        paths.responsibility_root, "train", train_sampler.get_index()
    )
    train_loader = DailyFeatureLoader(
        train_sampler,
        train_sidecar,
        device,
        shuffle=True,
        cache_on_gpu=cache_on_gpu,
    )
    train_description = sampler_description(train_sampler)
    history: List[Dict[str, Any]] = []
    threshold_triggered = False
    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        row = {
            "epoch": epoch,
            **train_epoch(
                model,
                optimizer,
                train_loader,
                direction_scale,
                selection_temperature(config, epoch),
                config.gradient_clip_value,
            ),
        }
        threshold_triggered = bool(row["TrainMSE"] <= config.threshold)
        row.update(
            {
                "threshold_reached": threshold_triggered,
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
        )
        history.append(row)
        atomic_csv(metric_dir / "train_history.csv", history)
        atomic_json(
            run_dir / "progress.json",
            {"status": "TRAINING", "latest": row, "updated_utc": utc_now()},
        )
        print(
            "dataset=%s seed=%d epoch=%d TrainMSE=%.9f"
            % (config.dataset, seed, epoch, row["TrainMSE"]),
            flush=True,
        )
        if mode == "formal" and threshold_triggered:
            break

    final_temperature = float(history[-1]["selection_temperature"])
    checkpoint = checkpoint_dir / "model_state.pt"
    torch.save(
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        checkpoint,
    )
    del train_loader, train_sidecar, train_sampler
    _release_memory()

    valid_sampler = load_sampler(paths.samplers["valid"])
    valid_sidecar = load_responsibility_segment(
        paths.responsibility_root, "valid", valid_sampler.get_index()
    )
    valid_prediction = predict_segment(
        model,
        valid_sampler,
        valid_sidecar,
        device,
        direction_scale,
        final_temperature,
        cache_on_gpu,
    )
    valid_prediction.to_pickle(prediction_dir / "valid_prediction.pkl")
    valid_dates = pd.DatetimeIndex(
        valid_prediction.index.get_level_values("datetime")
    )
    bridge_prediction = valid_prediction.loc[
        valid_dates == pd.Timestamp(config.bridge_date)
    ]
    if bridge_prediction.empty:
        raise RuntimeError(
            "bridge date %s is absent from validation prediction"
            % config.bridge_date
        )
    bridge_prediction.to_pickle(prediction_dir / "bridge_prediction.pkl")
    valid_description = sampler_description(valid_sampler)
    del valid_prediction, valid_sidecar, valid_sampler
    _release_memory()

    test_sampler = load_sampler(paths.samplers["test"])
    test_sidecar = load_responsibility_segment(
        paths.responsibility_root, "test", test_sampler.get_index()
    )
    test_prediction = predict_segment(
        model,
        test_sampler,
        test_sidecar,
        device,
        direction_scale,
        final_temperature,
        cache_on_gpu,
    )
    labels = labels_from_sampler(test_sampler)
    metrics, daily_rows, exclusions = daily_metrics(test_prediction, labels)
    test_prediction.to_pickle(prediction_dir / "test_prediction.pkl")
    atomic_csv(metric_dir / "test_daily_metrics.csv", daily_rows)
    atomic_json(metric_dir / "test_metric_exclusions.json", exclusions)
    signal = normalize_prediction(
        pd.concat([bridge_prediction, test_prediction]).sort_index()
    )
    signal.to_pickle(prediction_dir / "qlib_signal.pkl")
    test_description = sampler_description(test_sampler)

    result = {
        "protocol": PROTOCOL,
        "status": "DONE",
        "development_or_diagnostic": True,
        "independent_holdout": False,
        "dataset": config.dataset,
        "benchmark": config.benchmark,
        "seed": seed,
        "mode": mode,
        "model": "ResponsibilityModel",
        "method": "CRFR+stock-specific RRCA+uniform learning rate",
        "route_detached": False,
        "selection": {
            "rule": (
                "fixed_2_epoch_smoke"
                if mode == "smoke"
                else "first_mean_TrainMSE_lte_0.95_else_epoch150"
            ),
            "threshold": config.threshold,
            "max_epochs": max_epochs,
            "epochs_completed": len(history),
            "selected_epoch": len(history),
            "selected_train_mse": float(history[-1]["TrainMSE"]),
            "threshold_triggered": threshold_triggered,
            "stop_reason": (
                "fixed_2_epoch_smoke"
                if mode == "smoke"
                else (
                    "TrainMSE_lte_0.95" if threshold_triggered else "max_epoch_150"
                )
            ),
        },
        "metrics": {
            "IC": float(metrics["IC"]),
            "ICIR": None if metrics["ICIR"] is None else float(metrics["ICIR"]),
            "RankIC": float(metrics["RankIC"]),
            "RankICIR": (
                None
                if metrics["RankICIR"] is None
                else float(metrics["RankICIR"])
            ),
        },
        "metric_details": metrics,
        "parameter_counts": parameter_counts,
        "optimizer": optimizer_metadata,
        "data": {
            "train": train_description,
            "valid": valid_description,
            "test": test_description,
            "direction_scale": list(scale_payload["direction_scale"]),
            "label_processing": "daily_drop_top_bottom_2.5pct_then_sample_std_zscore",
        },
        "backtest_contract": {
            "strategy": "TopkDropoutStrategy",
            "topk": 30,
            "n_drop": 30,
            "headline": "excess_return_without_cost",
            "status": "NOT_RUN; use scripts/evaluate.py",
        },
        "artifacts": {
            "checkpoint": str(checkpoint),
            "test_prediction": str(prediction_dir / "test_prediction.pkl"),
            "qlib_signal": str(prediction_dir / "qlib_signal.pkl"),
        },
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(run_dir / "result.json", result)
    atomic_json(run_dir / "progress.json", {"status": "DONE", "updated_utc": utc_now()})
    atomic_text(run_dir / "DONE", result["finished_utc"] + "\n")
    del optimizer, model, test_sidecar, test_sampler, labels
    _release_memory()
    return result
