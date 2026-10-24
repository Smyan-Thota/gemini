"""CPU gradient equivalence for memory-bounded pairwise accumulation."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    import torch.nn.functional as functional
except ModuleNotFoundError:  # The standard-library-only environment can skip this autograd check.
    torch = None
    functional = None

from decision_factory.training.trainer import _accumulated_optimizer_update  # noqa: E402


@unittest.skipIf(torch is None, "PyTorch is required for the CPU autograd equivalence check")
class PairwiseGradientRecomputeTests(unittest.TestCase):
    def test_recomputed_pair_gradient_matches_joint_objective_for_mixed_units(self) -> None:
        keys = ["PERMIT", "BLOCK", "REVIEW"]
        rows = [
            {"target": 0, "keys": keys, "scale": 1.1, "bias": [0.10, -0.20, 0.05]},
            {"target": 1, "keys": keys, "scale": 0.7, "bias": [-0.10, 0.05, 0.20]},
            {"target": 0, "keys": keys, "scale": 0.5, "bias": [-0.05, 0.15, 0.00]},
            {"target": 1, "keys": keys, "scale": 1.3, "bias": [0.20, -0.10, 0.10]},
            {"target": 2, "keys": keys, "scale": 0.9, "bias": [0.00, 0.03, -0.02]},
        ]
        units = [(0, 1), (2, 3), (4,)]
        contrast_weight = 0.23
        contrast_margin = 0.8
        initial = torch.tensor([0.2, -0.1, 0.05], dtype=torch.float32)

        reference_parameter = torch.nn.Parameter(initial.clone())

        def logits_for(parameter, row):
            bias = torch.tensor(row["bias"], dtype=torch.float32)
            return parameter * row["scale"] + bias

        reference_logits = [logits_for(reference_parameter, row) for row in rows]
        ce_sum = sum(
            functional.cross_entropy(logits.unsqueeze(0), torch.tensor([row["target"]]))
            for logits, row in zip(reference_logits, rows)
        )
        contrast_terms = []
        for positive_index, negative_index in ((0, 1), (2, 3)):
            positive_margin = reference_logits[positive_index][0] - reference_logits[positive_index][1]
            negative_margin = reference_logits[negative_index][0] - reference_logits[negative_index][1]
            contrast_terms.append(functional.softplus(
                contrast_margin - (positive_margin - negative_margin)))
        joint_objective = (ce_sum / len(rows) + contrast_weight *
                           sum(contrast_terms) / len(contrast_terms))
        joint_objective.backward()
        expected_gradient = reference_parameter.grad.detach().clone()

        parameter = torch.nn.Parameter(initial.clone())

        class CaptureOptimizer:
            def __init__(self):
                self.steps = 0
                self.gradient = None

            def zero_grad(self, *, set_to_none=False):
                parameter.grad = None if set_to_none else torch.zeros_like(parameter)

            def step(self):
                self.steps += 1
                self.gradient = parameter.grad.detach().clone()

        optimizer = CaptureOptimizer()
        grad_modes = []
        parameter_snapshots = []

        def forward_one(row):
            self.assertEqual(optimizer.steps, 0, "an optimizer step occurred between forward passes")
            parameter_snapshots.append(parameter.detach().clone())
            grad_modes.append(torch.is_grad_enabled())
            return logits_for(parameter, row)

        metrics = _accumulated_optimizer_update(
            None, rows, units, None, torch, functional, optimizer,
            contrast_weight=contrast_weight, contrast_margin=contrast_margin,
            forward_one=forward_one,
        )

        self.assertEqual(optimizer.steps, 1)
        self.assertEqual(len(parameter_snapshots), len(rows) + 2 * len(contrast_terms))
        self.assertEqual(grad_modes.count(False), 2 * len(contrast_terms))
        self.assertEqual(grad_modes.count(True), len(rows))
        self.assertTrue(all(torch.equal(snapshot, initial) for snapshot in parameter_snapshots))
        self.assertTrue(torch.allclose(optimizer.gradient, expected_gradient, rtol=1e-6, atol=1e-7))
        self.assertAlmostEqual(metrics["total"], float(joint_objective.detach()), places=6)
        self.assertAlmostEqual(metrics["cross_entropy"], float((ce_sum / len(rows)).detach()), places=6)
        self.assertAlmostEqual(metrics["pairwise_contrast"],
                               float((sum(contrast_terms) / len(contrast_terms)).detach()), places=6)
        self.assertEqual(metrics["accumulated_examples"], 5.0)
        self.assertEqual(metrics["paired_units"], 2.0)
        self.assertEqual(metrics["model_forward_calls"], 9.0)
        self.assertEqual(metrics["pairwise_recompute_forward_calls"], 4.0)


if __name__ == "__main__":
    unittest.main()
