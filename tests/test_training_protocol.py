from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

from decision_factory.training.protocol import (
    assert_gemini_module_origins,
    balanced_accumulation_groups,
    balanced_source_indices,
    normalize_corpus,
    pair_indices,
)
from decision_factory.training.trainer import (
    _input_hash_rows,
    _accumulated_optimizer_update,
    _append_evaluation_metrics,
    _pilot_gate_metrics,
    _pilot_selection_key,
    _validate_warm_payload,
)


class TrainingProtocolTests(unittest.TestCase):
    def test_evaluation_metrics_are_durably_appended_without_prediction_rows(self) -> None:
        entry = {
            "step": 16, "checkpoint": "step-000016.pt", "checkpoint_sha256": "a" * 64,
            "correct": 120, "count": 152, "accuracy": 120 / 152,
            "pilot_metrics": {"all_gates_pass": False, "gates": {"safe": True}},
            "selection_key": [0, 1, 0.5], "predictions": [{"sensitive": "row detail"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            _append_evaluation_metrics(output, entry)
            _append_evaluation_metrics(output, entry)
            lines = (output / "evaluation-metrics.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        import json
        record = json.loads(lines[0])
        self.assertEqual(record["step"], 16)
        self.assertEqual(record["correct"], 120)
        self.assertNotIn("predictions", record)

    def test_join_keeps_requests_states_and_labels_separate(self) -> None:
        rows, labels = normalize_corpus(
            [
                {"scenario_id": "a", "source": "common", "model_state": {"schema": "s"}},
                {"scenario_id": "b", "source": "rare", "model_state": {"schema": "s"}},
            ],
            [
                {"scenario_id": "a", "guard_request": {"proposal": "a"}},
                {"scenario_id": "b", "guard_request": {"proposal": "b"}},
            ],
            [
                {"scenario_id": "a", "verdict": "PERMIT", "pair_id": "p", "group_id": "old96"},
                {"scenario_id": "b", "verdict": "BLOCK", "pair_id": "p"},
            ],
        )
        self.assertEqual([row["sample_id"] for row in rows], ["a", "b"])
        self.assertNotIn("verdict", rows[0])
        self.assertEqual(rows[0]["request"], {"proposal": "a"})
        self.assertEqual(labels["a"]["group_id"], "old96")
        self.assertEqual(pair_indices(rows, labels), [(0, 1)])

    def test_source_sampler_balances_buckets_and_is_repeatable(self) -> None:
        rows = [{"source": "common"}] * 3 + [{"source": "rare"}]
        first = balanced_source_indices(rows, seed=14, epoch=2)
        self.assertEqual(first, balanced_source_indices(rows, seed=14, epoch=2))
        counts = {source: 0 for source in ("common", "rare")}
        for index in first:
            counts[rows[index]["source"]] += 1
        self.assertEqual(counts, {"common": 3, "rare": 3})

    def test_accumulation_uses_singleton_forwards_and_one_optimizer_step(self) -> None:
        import torch
        import torch.nn.functional as functional

        parameter = torch.nn.Parameter(torch.tensor([0.2, -0.1, 0.05], dtype=torch.float32))
        base_optimizer = torch.optim.SGD([parameter], lr=0.05)

        class CountingOptimizer:
            def __init__(self, wrapped):
                self.wrapped = wrapped
                self.steps = 0
                self.gradient = None

            def zero_grad(self, **kwargs):
                return self.wrapped.zero_grad(**kwargs)

            def step(self):
                self.steps += 1
                self.gradient = parameter.grad.detach().clone()
                return self.wrapped.step()

        optimizer = CountingOptimizer(base_optimizer)
        rows = [{"target": 0, "keys": ["PERMIT", "BLOCK", "REVIEW"], "scale": 1 / 24},
                {"target": 1, "keys": ["PERMIT", "BLOCK", "REVIEW"], "scale": 2 / 24}]
        rows.extend({"target": index % 3, "keys": ["PERMIT", "BLOCK", "REVIEW"],
                     "scale": index / 24}
                    for index in range(3, 25))
        calls = []

        def forward_one(row):
            calls.append(row)
            return parameter * row["scale"]

        reference = torch.nn.Parameter(before := parameter.detach().clone())
        reference_logits = [reference * row["scale"] for row in rows]
        reference_ce = sum(functional.cross_entropy(logits.unsqueeze(0),
            torch.tensor([row["target"]])) for logits, row in zip(reference_logits, rows)) / 24
        permit_margin = reference_logits[0][0] - reference_logits[0][1]
        block_margin = reference_logits[1][0] - reference_logits[1][1]
        reference_loss = reference_ce + 0.1 * functional.softplus(1.0 - (permit_margin - block_margin))
        reference_loss.backward()
        expected_gradient = reference.grad.detach().clone()

        metrics = _accumulated_optimizer_update(
            None, rows, [(0, 1)] + [(index,) for index in range(2, 24)],
            None, torch, functional, optimizer, contrast_weight=0.1,
            contrast_margin=1.0, forward_one=forward_one)
        self.assertEqual(len(calls), 26)
        self.assertEqual(optimizer.steps, 1)
        self.assertEqual(metrics["accumulated_examples"], 24.0)
        self.assertEqual(metrics["pairwise_recompute_forward_calls"], 2.0)
        self.assertTrue(torch.allclose(optimizer.gradient, expected_gradient, rtol=1e-6, atol=1e-7))
        self.assertFalse(torch.equal(before, parameter.detach()))

    def test_corrected_schedule_consumes_rows_once_in_balanced_updates(self) -> None:
        rows = []
        labels = {}
        for source in ("fresh", "retained"):
            for index in range(24):
                sample_id = f"{source}-{index}"
                rows.append({"sample_id": sample_id, "source": source})
                labels[sample_id] = {"verdict": "PERMIT" if index % 2 == 0 else "BLOCK"}
        groups = balanced_accumulation_groups(rows, labels, seed=17, examples_per_update=24,
                                               use_pairs=False)
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(sum(len(unit) for unit in group) == 24 for group in groups))
        for group in groups:
            counts = {source: sum(1 for unit in group for index in unit
                                  if rows[index]["source"] == source)
                      for source in ("fresh", "retained")}
            self.assertEqual(counts, {"fresh": 12, "retained": 12})

    def test_shadowed_projection_module_fails_closed(self) -> None:
        module = types.ModuleType("alpha_bind_v1")
        module.__file__ = str(Path(__file__).resolve().parent / "shadow.py")
        sys.modules["alpha_bind_v1"] = module
        try:
            with self.assertRaisesRegex(RuntimeError, "outside Gemini source"):
                assert_gemini_module_origins(Path(__file__).resolve().parents[1] / "src")
        finally:
            del sys.modules["alpha_bind_v1"]

    def test_input_provenance_keeps_compact_hashes_only(self) -> None:
        rows = _input_hash_rows([{
            "sample_id": "a", "source": "read", "state": {"large": "state"},
            "request": {"large": "request"}, "attention_matrix": object(),
            "state_sha256": "a" * 64, "state_text_sha256": "b" * 64,
            "request_sha256": "c" * 64, "label_sha256": "0" * 64,
            "encoded_sha256": "d" * 64,
            "input_ids_sha256": "e" * 64, "attention_mask_sha256": "f" * 64,
            "question_sha256": "1" * 64, "options_sha256": "2" * 64,
        }])
        self.assertEqual(set(rows[0]), {
            "sample_id", "source", "state_sha256", "state_text_sha256", "request_sha256",
            "label_sha256", "encoded_sha256", "input_ids_sha256", "attention_mask_sha256",
            "question_sha256", "options_sha256",
        })
        self.assertNotIn("attention_matrix", rows[0])

    def test_pilot_gates_and_selection_key(self) -> None:
        rows = []
        for index in range(32):
            rows.append({"group_id": "old96", "pair_id": f"p{index}",
                         "verdict": "PERMIT", "prediction": "PERMIT"})
            rows.append({"group_id": "old96", "pair_id": f"p{index}",
                         "verdict": "BLOCK", "prediction": "BLOCK"})
        rows.extend({"group_id": "old96", "verdict": "REVIEW", "prediction": "REVIEW"}
                    for _ in range(32))
        rows.extend({"group_id": "actual56", "verdict": "PERMIT", "prediction": "PERMIT"}
                    for _ in range(52))
        rows.extend({"group_id": "actual56", "verdict": "BLOCK", "prediction": "BLOCK"}
                    for _ in range(4))
        metrics = _pilot_gate_metrics(rows)
        self.assertTrue(metrics["all_gates_pass"])
        self.assertEqual(metrics["strict_pairs"]["correct_pairs"], 32)
        self.assertEqual(metrics["actual56"]["blocks_correct"], 4)
        self.assertGreater(_pilot_selection_key(metrics, 16), _pilot_selection_key(metrics, 32))
        rows[-1]["prediction"] = "PERMIT"
        failed = _pilot_gate_metrics(rows)
        self.assertFalse(failed["all_gates_pass"])
        self.assertEqual(failed["actual56"]["unsafe_permits_on_blocks"], 1)

    def test_only_exact_pinned_v9_prior_is_a_legacy_warm_start(self) -> None:
        bev = SimpleNamespace(MODEL_ID="id", MODEL_REVISION="rev", MODEL_SHA256="a" * 64)
        sequence = SimpleNamespace(ALPHA_BIND_V2_SHA256="b" * 64,
                                   NATIVE_STATE_V2_SHA256="e" * 64)
        config = {"model_state_manifest_sha256": "c" * 64,
                  "model_state_independent_audit_sha256": "d" * 64}
        payload = {
            "schema": "bev_lora_head_checkpoint/2", "base_model_id": "id",
            "base_revision": "rev", "base_weight_sha256": "a" * 64,
            "training_input_mode": "stored_alpha_v2_state_no_runtime_retransform",
            "candidate_training_contract": "sequence-guard-v9-native-v2-full-depth-balanced-source-sequence-only",
            "model_state_schema": "sequence_guard_native_state_alpha/2",
            "model_state_transform_sha256": "b" * 64,
            "model_state_manifest_sha256": "254026dbcc2f8118d78e44c6dbaf5957bb10315f392416e02a131799a6a82359",
            "model_state_independent_audit_sha256": None, "selected_step": 64,
        }
        self.assertEqual(_validate_warm_payload(
            payload, "ec3b8ef8fddc4b5560e25149e5d0c98999b6046e58c187e116cf26bc8aeab7c2",
            bev, sequence, config), "unpromoted_v9_prior")
        with self.assertRaisesRegex(ValueError, "alpha-v2 input contract"):
            _validate_warm_payload(payload, "0" * 64, bev, sequence, config)


if __name__ == "__main__":
    unittest.main()
