"""Independent cheap checks for frozen-feature extraction and runtime scoring."""
from __future__ import annotations

import hashlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from decision_factory.runtime_guard import frozen_readout as readout


class FrozenReadoutContractTests(unittest.TestCase):
    def test_extraction_preserves_attention_and_answer_option_positions(self):
        batch = {"input_ids": torch.arange(10).reshape(1, 10),
                 "position_ids": torch.tensor([[0, 1, 2, 3, 4, 5, 3, 4, 5, 3]]),
                 "attention_mask": torch.ones((1, 1, 10, 10), dtype=torch.bool).tril(),
                 "choice_read_idx": torch.tensor([[4, 7, 9]]), "answer_end_idx": torch.tensor([2]),
                 "choice_mask": torch.ones((1, 3), dtype=torch.bool), "task_type": torch.tensor([0])}
        middle = torch.arange(20, dtype=torch.float32).reshape(1, 10, 2)
        last = middle + 100
        seen = {}
        def backbone(**kwargs):
            seen.update(kwargs)
            return types.SimpleNamespace(hidden_states=(middle * 0, middle, last), last_hidden_state=last)
        def head(options, answer, mask, task):
            self.assertTrue(torch.equal(options, last[:, [4, 7, 9]]))
            self.assertTrue(torch.equal(answer, last[:, 2]))
            self.assertTrue(torch.equal(mask, batch["choice_mask"]))
            return torch.tensor([[1., 2., 3.]])
        decider = types.SimpleNamespace(encoder=types.SimpleNamespace(collate=lambda _: batch),
                                        network=types.SimpleNamespace(backbone=backbone, head=head))
        features, logits = readout.extract_features(decider, {}, device="cpu")
        expected = torch.cat((middle[0, 2], last[0, 2], middle[0, [4, 7, 9]].flatten(), last[0, [4, 7, 9]].flatten()))
        self.assertTrue(torch.equal(features, expected))
        self.assertTrue(torch.equal(logits, torch.tensor([1., 2., 3.])))
        self.assertTrue(torch.equal(seen["attention_mask"], batch["attention_mask"]))
        self.assertTrue(torch.equal(seen["position_ids"], batch["position_ids"]))
        self.assertFalse(seen["use_cache"])

    def test_overlong_scene_is_rejected_before_encoder_can_truncate_it(self):
        encoder = types.SimpleNamespace(tok=types.SimpleNamespace(encode=lambda _: [1, 2, 3]), max_state_tokens=2,
                                        encode=lambda *_: self.fail("encoder should not receive overflowing scene"))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            readout.encode_scene(types.SimpleNamespace(encoder=encoder), {"fact": "too long"})

    def test_class_order_mismatch_is_rejected(self):
        encoder = types.SimpleNamespace(tok=types.SimpleNamespace(encode=lambda _: [1]), max_state_tokens=2,
                                        encode=lambda *_: {"keys": ["BLOCK", "PERMIT", "REVIEW"]})
        with self.assertRaisesRegex(ValueError, "class order"):
            readout.encode_scene(types.SimpleNamespace(encoder=encoder), {"fact": "ok"})

    def test_low_permit_score_can_only_defer_never_make_a_new_permit(self):
        model = object.__new__(readout.FrozenBevReadout)
        model.columns = [0, 1, 2]
        model.mean, model.scale = torch.zeros(3), torch.ones(3)
        model.layers = [{"weight": torch.eye(3), "bias": torch.zeros(3)}]
        model.temperature, model.permit_threshold, model.block_threshold = 1., .9, 0.
        verdict, scores = model.predict_features(torch.tensor([1., 0., 0.]))
        self.assertEqual(verdict, "REVIEW")
        self.assertAlmostEqual(sum(scores.values()), 1., places=6)
        self.assertGreater(scores["PERMIT"], scores["BLOCK"])

    def test_bad_artifact_provenance_is_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pt"
            torch.save({"schema": "unknown"}, path)
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(readout.bev, "_load_real_decider", side_effect=AssertionError("model loaded")):
                with self.assertRaisesRegex(ValueError, "provenance"):
                    readout.load_readout(path, checksum, device="cpu")


if __name__ == "__main__":
    unittest.main()
