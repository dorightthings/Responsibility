"""Small deterministic checks for the released within-window mechanism memory."""

from __future__ import annotations

import unittest

import torch

from responsibility.rrca import MechanismGroupedContext, ema_contexts


class MechanismMemoryTest(unittest.TestCase):
    def test_recurrence_and_mechanism_specific_gradients(self) -> None:
        contexts = torch.tensor([
            [[1.0], [3.0], [5.0]],
            [[2.0], [6.0], [10.0]],
            [[3.0], [9.0], [15.0]],
            [[4.0], [12.0], [20.0]],
        ], requires_grad=True)
        raw = torch.zeros(4, requires_grad=True)
        output = ema_contexts(contexts, raw)
        expected = torch.tensor([
            [[1.0], [2.0], [3.5]],
            [[2.0], [4.0], [7.0]],
            [[3.0], [6.0], [10.5]],
            [[4.0], [8.0], [14.0]],
        ])
        self.assertTrue(torch.equal(output, expected))
        output[:, -1].sum().backward()
        self.assertEqual(torch.count_nonzero(raw.grad).item(), 4)
        self.assertEqual(torch.unique(raw.grad).numel(), 4)
        self.assertTrue(torch.isfinite(contexts.grad).all().item())

    def test_causal_and_no_cross_window_state(self) -> None:
        torch.manual_seed(19)
        context = MechanismGroupedContext(d_model=8, adapter_rank=3).eval()
        self.assertTrue(torch.equal(context.memory_coefficients, torch.full((4,), 0.5)))
        hidden = torch.randn(5, 8, 8)
        responsibility = torch.rand(5, 4)
        valid = torch.ones(5, dtype=torch.bool)
        with torch.no_grad():
            original = context(hidden, responsibility, valid)
            changed = hidden.clone()
            changed[:, 4:] += torch.randn_like(changed[:, 4:])
            later_changed = context(changed, responsibility, valid)
            self.assertTrue(torch.equal(original[:, :4], later_changed[:, :4]))
            self.assertTrue(torch.equal(original, context(hidden, responsibility, valid)))

    def test_rejects_wrong_memory_shape(self) -> None:
        with self.assertRaises(ValueError):
            ema_contexts(torch.zeros(4, 8, 3), torch.zeros(1))
        with self.assertRaises(ValueError):
            ema_contexts(torch.zeros(4, 0, 3), torch.zeros(4))


if __name__ == "__main__":
    unittest.main()
