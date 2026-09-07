"""CPU checks for the restored two-group Adam configuration."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from responsibility.training.config import load_experiment_config
from responsibility.training.engine import _model, build_optimizer, optimizer_metadata


ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    ("short_csi300", "csi300.yaml"),
    ("short_csi800_direct", "csi800_direct.yaml"),
)
LR_FIELDS = {
    "base_learning_rate": 1e-5,
    "crfr_learning_rate": 1e-4,
    "rrca_adapter_learning_rate": 1e-5,
    "rrca_condition_learning_rate": 1e-4,
}


def _load_payload(payload: dict):
    with patch(
        "responsibility.training.config._read_mapping", return_value=payload
    ):
        return load_experiment_config(ROOT / "configs/csi300.yaml")


class GroupedLearningRateConfigTest(unittest.TestCase):
    def test_defaults_match_released_yaml(self) -> None:
        for dataset, filename in DATASETS:
            with self.subTest(dataset=dataset):
                for config in (
                    _load_payload({"dataset": dataset}),
                    load_experiment_config(ROOT / "configs" / filename),
                ):
                    self.assertEqual(config.dataset, dataset)
                    for name, value in LR_FIELDS.items():
                        self.assertEqual(getattr(config, name), value)
                    self.assertNotIn("learning_rate", config.to_dict())

    def test_rejects_uniform_release_field(self) -> None:
        for location in ("root", "training"):
            payload = {"dataset": "short_csi300", "training": {}}
            target = payload if location == "root" else payload["training"]
            target["learning_rate"] = 2e-5
            with self.subTest(location=location), self.assertRaises(ValueError):
                _load_payload(payload)

    def test_rejects_changed_or_nonfinite_group_rates(self) -> None:
        for field in LR_FIELDS:
            for value in (0.0, -1e-5, 3e-5, float("nan"), float("inf")):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    _load_payload({
                        "dataset": "short_csi300",
                        "training": {field: value},
                    })


class GroupedOptimizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def test_exact_groups_and_complete_coverage(self) -> None:
        for dataset, filename in DATASETS:
            with self.subTest(dataset=dataset):
                config = load_experiment_config(ROOT / "configs" / filename)
                model = _model(config, seed=7, device=torch.device("cpu"))
                optimizer, counts = build_optimizer(model, config)
                self.assertIsInstance(optimizer, torch.optim.Adam)
                self.assertEqual(len(optimizer.param_groups), 2)
                fast_names = set(model.g1_module_parameter_names()) | set(
                    model.condition_parameter_names()
                )
                named = dict(model.named_parameters())
                all_ids = []
                for index, group in enumerate(optimizer.param_groups):
                    self.assertEqual(group["lr"], (1e-5, 1e-4)[index])
                    expected = {
                        id(parameter) for name, parameter in named.items()
                        if (name in fast_names) == (index == 1)
                    }
                    actual = [id(p) for p in group["params"]]
                    self.assertEqual(set(actual), expected)
                    all_ids.extend(actual)
                self.assertEqual(len(all_ids), len(set(all_ids)))
                self.assertEqual(set(all_ids), {id(p) for p in model.parameters()})
                self.assertEqual(counts["total"], 846229)
                self.assertEqual(counts["rrca_condition"], 1)
                self.assertNotIn("rrca_memory", counts)
                self.assertFalse(any("raw_memory" in name for name in named))
                self.assertEqual(
                    sum(value for key, value in counts.items() if key != "total"),
                    counts["total"],
                )
                recorded = optimizer_metadata(optimizer)
                self.assertEqual(recorded["parameter_group_count"], 2)
                self.assertEqual(
                    [group["learning_rate"] for group in recorded["groups"]],
                    [1e-5, 1e-4],
                )
                self.assertEqual(
                    sum(group["parameter_count"] for group in recorded["groups"]),
                    846229,
                )

    def test_two_updates_match_original_grouped_adam(self) -> None:
        config = load_experiment_config(ROOT / "configs/csi300.yaml")
        model = _model(config, seed=7, device=torch.device("cpu"))
        reference = copy.deepcopy(model)
        optimizer, _ = build_optimizer(model, config)
        fast = set(reference.g1_module_parameter_names()) | set(
            reference.condition_parameter_names()
        )
        direct = torch.optim.Adam([
            {"params": [p for n, p in reference.named_parameters() if n not in fast],
             "lr": 1e-5},
            {"params": [p for n, p in reference.named_parameters() if n in fast],
             "lr": 1e-4},
        ])
        for step in range(2):
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
