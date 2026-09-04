"""CPU smoke tests for the portable responsibility model."""

from __future__ import annotations

import unittest

import torch

from responsibility import ResponsibilityModel, build_model


def _model_kwargs() -> dict:
    return {
        "d_feat": 158,
        "d_model": 256,
        "t_nhead": 4,
        "s_nhead": 2,
        "T_dropout_rate": 0.5,
        "S_dropout_rate": 0.5,
        "gate_input_start_index": 158,
        "gate_input_end_index": 221,
        "beta": 5.0,
        "rule_seed": 7,
        "selection_temperature": 1.0,
        "alpha_initial": 0.1,
        "rho_initial": 0.1,
        "adapter_rank": 32,
        "gate_mode": "shared",
    }


class ResponsibilityModelTest(unittest.TestCase):
    def test_random_cpu_forward_and_backward(self) -> None:
        torch.manual_seed(11)
        model = build_model(**_model_kwargs())
        self.assertIsInstance(model, ResponsibilityModel)
        self.assertEqual(next(model.parameters()).device.type, "cpu")

        stocks, steps = 5, 8
        x = torch.randn(stocks, steps, 221)
        stock_state = torch.randn(stocks, 14)
        rule_components = torch.randn(stocks, 8)
        rule_components[0].zero_()
        direction_scale = torch.ones(4)

        prediction, auxiliary = model(
            x,
            stock_state,
            rule_components,
            direction_scale,
            return_aux=True,
        )

        self.assertEqual(prediction.shape, (stocks,))
        self.assertTrue(torch.isfinite(prediction).all().item())
        self.assertEqual(auxiliary["gate_weights"].shape, (stocks, 158))
        self.assertEqual(auxiliary["context_source_weights"].shape, (stocks, 4))
        self.assertTrue(
            torch.allclose(
                auxiliary["gate_weights"].sum(dim=1),
                torch.full((stocks,), 158.0),
                atol=1e-4,
                rtol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                auxiliary["context_source_weight_sums"],
                torch.ones(4),
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertEqual(
            torch.count_nonzero(
                auxiliary["context_target_responsibility"][0]
            ).item(),
            0,
        )

        prediction.square().mean().backward()
        self.assertIsNotNone(model.feature_gate.trans.weight.grad)
        self.assertIsNotNone(model.mechanism_preference.grad)
        self.assertIsNotNone(model.mechanism_context.rho_logit.grad)

    def test_fusion_disabled_matches_unmodified_backbone_path(self) -> None:
        torch.manual_seed(13)
        model = build_model(**_model_kwargs()).eval()
        stocks, steps = 4, 8
        x = torch.randn(stocks, steps, 221)
        stock_state = torch.randn(stocks, 14)
        rule_components = torch.randn(stocks, 8)
        direction_scale = torch.ones(4)

        with torch.no_grad():
            prediction = model(
                x,
                stock_state,
                rule_components,
                direction_scale,
                fusion_enabled=False,
            )
            src = x[:, :, :158]
            market = x[:, -1, 158:221]
            gate_weights = model.feature_gate(market)
            expected = model.layers(src * gate_weights.unsqueeze(1)).squeeze(-1)

        self.assertTrue(torch.equal(prediction, expected))


if __name__ == "__main__":
    unittest.main()
