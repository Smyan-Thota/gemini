"""Structured generation boundary and completion-loss alignment checks."""
from __future__ import annotations

import copy
import json
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from decision_factory.runtime_guard.authorization_actor import leading_json_command
from decision_factory.recovery.core import _strict_decision


class AuthorizationActorTests(unittest.TestCase):
    def test_only_first_complete_command_is_consumed_and_tail_is_recorded(self):
        raw = ' {"type":"ACT","action":"opaque-id"},{"type":"STOP"}'
        command, boundary = leading_json_command(raw)
        self.assertEqual(_strict_decision(command)[0], {"type": "ACT", "action": "opaque-id"})
        self.assertEqual(raw[boundary["start_character"]:boundary["end_character"]], command)
        self.assertGreater(boundary["ignored_trailing_characters"], 0)

    def test_prose_prefix_is_not_searched_for_an_embedded_command(self):
        for raw in ('Here: {"type":"ACT","action":"x"}', '```json\n{"type":"STOP"}\n```', ''):
            with self.assertRaises(ValueError):
                leading_json_command(raw)

    def test_invalid_first_object_cannot_be_replaced_by_valid_later_object(self):
        command, _ = leading_json_command('{"type":"INVALID"},{"type":"STOP"}')
        self.assertIsNone(_strict_decision(command)[0])

    def test_incomplete_json_is_not_repaired_or_invented(self):
        with self.assertRaises(ValueError):
            leading_json_command('{"type":"ACT","action":"unfinished')

    def test_response_only_loss_and_gradients_match_full_masked_causal_loss(self):
        import torch
        from decision_factory.training.actor_repair import response_loss
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(8, 4)
                self.output = torch.nn.Linear(4, 8)
            def forward(self, input_ids, attention_mask, use_cache, logits_to_keep=0):
                # Causal hidden states depend on prefixes; the response's first
                # token must be predicted from the last prompt position.
                logits = self.output(self.embedding(input_ids).cumsum(1))
                return types.SimpleNamespace(logits=logits[:, -logits_to_keep:] if logits_to_keep else logits)
        torch.manual_seed(7)
        short = Tiny()
        full = copy.deepcopy(short)
        encoded = ([1, 2, 3, 4, 5], [4, 5, 6])
        loss = response_loss(short, encoded, device="cpu")
        loss.backward()
        ids = torch.tensor([encoded[0]])
        logits = full(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0]
        reference = torch.nn.functional.cross_entropy(logits, torch.tensor([-100, -100, 4, 5, 6]))
        reference.backward()
        self.assertTrue(torch.allclose(loss, reference, atol=1e-7))
        for parameter, expected in zip(short.parameters(), full.parameters()):
            self.assertTrue(torch.allclose(parameter.grad, expected.grad, atol=1e-6))

    def test_weighted_loss_normalizes_by_content_weight_sum(self):
        import torch
        from decision_factory.training.actor_repair import response_loss
        class Fixed(torch.nn.Module):
            def forward(self, **kwargs):
                return types.SimpleNamespace(logits=torch.tensor([[[2., 0.], [0., 2.], [1., 0.]]]))
        encoded = ([1, 2, 3], [0, 0, 1], [.25, 8., 1.], [False, True, False])
        loss, metrics = response_loss(Fixed(), encoded, device="cpu", return_metrics=True)
        ce = torch.nn.functional.cross_entropy(Fixed()().logits[0], torch.tensor([0, 0, 1]), reduction="none")
        self.assertAlmostEqual(float(loss), float((ce * torch.tensor([.25, 8., 1.])).sum() / 9.25), places=6)
        self.assertFalse(metrics["teacher_forced_critical_exact"])


if __name__ == "__main__":
    unittest.main()
