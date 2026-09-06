"""Small CPU checks for the released single-learning-rate configuration."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from responsibility.training.config import load_experiment_config
from responsibility.training.engine import _model, build_optimizer


ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    ("short_csi300", "csi300.yaml", 2e-5),
    ("short_csi800_direct", "csi800_direct.yaml", 1e-5),
)
LEGACY_LR_FIELDS = (
    "base_learning_rate",
    "crfr_learning_rate",
    "rrca_adapter_learning_rate",
    "rrca_condition_learning_rate",
)


def _load_payload(payload: dict):
    # Keep path validation real while avoiding temporary configuration files.
    with patch(
        "responsibility.training.config._read_mapping", return_value=payload
    ):
        return load_experiment_config(ROOT / "configs/csi300.yaml")


class UniformLearningRateConfigTest(unittest.TestCase):
    def test_dataset_defaults_and_released_yaml_configs(self) -> None:
        for dataset, filename, expected_lr in DATASETS:
            with self.subTest(dataset=dataset):
                default = _load_payload({"dataset": dataset})
                released = load_experiment_config(ROOT / "configs" / filename)
                self.assertEqual(default.learning_rate, expected_lr)
                self.assertEqual(released.learning_rate, expected_lr)
                self.assertEqual(released.dataset, dataset)
                for config in (default, released):
                    self.assertEqual(config.to_dict()["learning_rate"], expected_lr)
                    for field in LEGACY_LR_FIELDS:
                        self.assertNotIn(field, config.to_dict())

    def test_accepts_explicit_positive_finite_learning_rate(self) -> None:
        for dataset, _, _ in DATASETS:
            with self.subTest(dataset=dataset):
                config = _load_payload({
                    "dataset": dataset,
                    "training": {"learning_rate": 3e-5},
                })
                self.assertEqual(config.learning_rate, 3e-5)

    def test_rejects_nonfinite_and_nonpositive_learning_rates(self) -> None:
        for value in (float("nan"), float("inf"), -float("inf"), 0.0, -1e-5):
            with self.subTest(learning_rate=value):
                with self.assertRaises(ValueError):
                    _load_payload({
                        "dataset": "short_csi300",
                        "training": {"learning_rate": value},
                    })

    def test_rejects_legacy_lr_fields_at_root_and_in_training(self) -> None:
        for field in LEGACY_LR_FIELDS:
            for location in ("root", "training"):
                with self.subTest(field=field, location=location):
                    payload = {
                        "dataset": "short_csi300",
                        "training": {"learning_rate": 2e-5},
                    }
                    target = payload if location == "root" else payload["training"]
                    target[field] = 1e-5
                    with self.assertRaises(ValueError):
                        _load_payload(payload)


class UniformOptimizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def test_single_group_covers_full_model_once(self) -> None:
        for dataset, _, expected_lr in DATASETS:
            with self.subTest(dataset=dataset):
                config = _load_payload({"dataset": dataset})
                model = _model(config, seed=7, device=torch.device("cpu"))
                optimizer, counts = build_optimizer(model, config)
                self.assertIsInstance(optimizer, torch.optim.Adam)
                self.assertEqual(len(optimizer.param_groups), 1)
                group = optimizer.param_groups[0]
                self.assertEqual(group["lr"], expected_lr)
                actual_ids = [id(parameter) for parameter in group["params"]]
                expected_ids = {
                    id(parameter) for parameter in model.parameters()
                    if parameter.requires_grad
                }
                self.assertEqual(set(actual_ids), expected_ids)
                self.assertEqual(len(actual_ids), len(expected_ids))
                self.assertEqual(sum(p.numel() for p in group["params"]), 846229)
                self.assertEqual(counts["total"], 846229)
                self.assertEqual(set(counts), {
                    "total", "backbone", "crfr", "rrca_adapters", "rrca_condition",
                })
                self.assertEqual(
                    sum(value for key, value in counts.items() if key != "total"),
                    counts["total"],
                )

                # Future experiments may freeze a parameter; it must not enter Adam.
                model.alpha_logit.requires_grad_(False)
                frozen_optimizer, _ = build_optimizer(model, config)
                actual_ids = [
                    id(parameter)
                    for entry in frozen_optimizer.param_groups
                    for parameter in entry["params"]
                ]
                expected_ids = {
                    id(parameter) for parameter in model.parameters()
                    if parameter.requires_grad
                }
                self.assertEqual(set(actual_ids), expected_ids)
                self.assertEqual(len(actual_ids), len(expected_ids))
                self.assertNotIn(id(model.alpha_logit), actual_ids)

    def test_two_updates_equal_direct_torch_adam(self) -> None:
        for dataset, _, _ in DATASETS:
            with self.subTest(dataset=dataset):
                config = _load_payload({"dataset": dataset})
                model = _model(config, seed=7, device=torch.device("cpu"))
                reference = copy.deepcopy(model)
                optimizer, _ = build_optimizer(model, config)
                direct = torch.optim.Adam(
                    reference.parameters(), lr=config.learning_rate
                )
                for step in range(2):
                    # Identical synthetic gradients exercise every parameter without
                    # another full-model forward/backward smoke test.
                    for index, (parameter, expected) in enumerate(zip(
                        model.parameters(), reference.parameters()
                    )):
                        value = ((index + 3 * step) % 11 - 5) * 0.01
                        parameter.grad = torch.full_like(parameter, value)
                        expected.grad = parameter.grad.clone()
                    optimizer.step()
                    direct.step()
                    for (name, parameter), (_, expected) in zip(
                        model.named_parameters(), reference.named_parameters()
                    ):
                        self.assertTrue(torch.equal(parameter, expected), name)
                        self.assertTrue(torch.equal(
                            optimizer.state[parameter]["exp_avg"],
                            direct.state[expected]["exp_avg"],
                        ), name)
                        self.assertTrue(torch.equal(
                            optimizer.state[parameter]["exp_avg_sq"],
                            direct.state[expected]["exp_avg_sq"],
                        ), name)


if __name__ == "__main__":
    unittest.main()
