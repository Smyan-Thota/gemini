"""Candidate assessor contract checks; these tests never load model weights."""
from __future__ import annotations

import copy
import hashlib
import sys
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_factory.runtime_guard import bev, sequence_v2  # noqa: E402
from decision_factory.runtime_guard.sequence_candidate import (  # noqa: E402
    CandidateAssessmentError,
    _validate_checkpoint_metadata,
    assess_candidate,
    release_candidate_cache,
)


def _payload() -> dict:
    return {
        "schema": "bev_lora_head_checkpoint/2",
        "base_model_id": bev.MODEL_ID,
        "base_revision": bev.MODEL_REVISION,
        "base_weight_sha256": bev.MODEL_SHA256,
        "training_input_mode": "production_native_v2_token_stream_singleton_fp32",
        "model_state_schema": "sequence_guard_native_state_alpha/2",
        "model_state_encoder_sha256": sequence_v2.NATIVE_STATE_V2_SHA256,
        "model_state_transform_sha256": sequence_v2.ALPHA_BIND_V2_SHA256,
        "model_state_manifest_sha256": "1" * 64,
        "model_state_independent_audit_sha256": "2" * 64,
        "selected_step": 64,
    }


def _request() -> dict:
    return {
        "schema": "bev_guard_request/1",
        "user_request": {"text": "Upload the approved report", "trust": "trusted"},
        "host_policy": {
            "allowed_operations": ["upload_file"],
            "max_operations": 2,
            "max_write_bytes": 1024,
            "workspace_root": "/workspace",
            "tool_schemas": {"upload_file": {"required": ["path", "recipient"]}},
            "workflow_rules": {"rule_type": "recipient_allowlist",
                               "recipient_directory_status": "verified",
                               "approved_recipients": ["archive-good"]},
        },
        "proposal": {"tool_name": "upload_file",
                     "arguments": {"path": "/workspace/report.txt", "recipient": "archive-good"}},
        "prior_actions": [],
        "observations": [],
    }


class SequenceCandidateContractTests(unittest.TestCase):
    def setUp(self):
        self._candidate_cache = dict(bev._candidate_deciders)
        bev._candidate_deciders.clear()

    def tearDown(self):
        bev._candidate_deciders.clear()
        bev._candidate_deciders.update(self._candidate_cache)

    def test_standard_alpha_v2_candidate_metadata_is_strict(self):
        payload = _payload()
        metadata, role, eligible = _validate_checkpoint_metadata(
            payload, checkpoint_sha256="3" * 64, allow_unpromoted_v9_prior=False)
        self.assertEqual(metadata["model_state_schema"], "sequence_guard_native_state_alpha/2")
        self.assertEqual(role, "candidate")
        self.assertTrue(eligible)

        bad = copy.deepcopy(payload)
        bad["model_state_encoder_sha256"] = "0" * 64
        with self.assertRaises(CandidateAssessmentError) as caught:
            _validate_checkpoint_metadata(bad, checkpoint_sha256="3" * 64,
                                          allow_unpromoted_v9_prior=False)
        self.assertEqual(caught.exception.error_code, "native_encoder_mismatch")

    def test_only_exact_v9_step64_is_available_as_unpromoted_prior(self):
        payload = {
            "schema": "bev_lora_head_checkpoint/2",
            "base_model_id": bev.MODEL_ID,
            "base_revision": bev.MODEL_REVISION,
            "base_weight_sha256": bev.MODEL_SHA256,
            "training_input_mode": "stored_alpha_v2_state_no_runtime_retransform",
            "candidate_training_contract": "sequence-guard-v9-native-v2-full-depth-balanced-source-sequence-only",
            "model_state_schema": "sequence_guard_native_state_alpha/2",
            "model_state_transform_sha256": sequence_v2.ALPHA_BIND_V2_SHA256,
            "model_state_manifest_sha256": "254026dbcc2f8118d78e44c6dbaf5957bb10315f392416e02a131799a6a82359",
            "model_state_independent_audit_sha256": None,
            "selected_step": 64,
        }
        _, role, eligible = _validate_checkpoint_metadata(
            payload,
            checkpoint_sha256="ec3b8ef8fddc4b5560e25149e5d0c98999b6046e58c187e116cf26bc8aeab7c2",
            allow_unpromoted_v9_prior=True,
        )
        self.assertEqual(role, "unpromoted_warm_start_prior")
        self.assertFalse(eligible)

        with self.assertRaises(CandidateAssessmentError):
            _validate_checkpoint_metadata(payload, checkpoint_sha256="4" * 64,
                                          allow_unpromoted_v9_prior=True)

    def test_cpu_rejection_is_typed_and_happens_before_checkpoint_access(self):
        result = assess_candidate(_request(), checkpoint_path=ROOT / "does-not-exist.pt",
                                  expected_checkpoint_sha256="a" * 64, device="cpu")
        evidence = result["evidence"]
        self.assertEqual(result["disposition"], "REVIEW")
        self.assertEqual(evidence["failure_stage"], "device_validation")
        self.assertEqual(evidence["error_code"], "unsupported_device")
        self.assertEqual(evidence["requested_device"], "cpu")
        self.assertIsNone(evidence["effective_device"])
        self.assertIsNotNone(evidence["request_sha256"])

    def test_cache_release_preserves_training_network_and_evicts_eval_network(self):
        class Network:
            def __init__(self):
                self.devices = []

            def to(self, *, device):
                self.devices.append(device)
                return self

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.pt"
            path.write_bytes(b"candidate")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            training_network = Network()
            bev._candidate_deciders[digest] = SimpleNamespace(network=training_network)
            retained = release_candidate_cache(path, expected_checkpoint_sha256=digest,
                                               preserve_network=training_network)
            self.assertEqual(retained["status"], "preserved_training_model")
            self.assertEqual(training_network.devices, [])
            self.assertIn(digest, bev._candidate_deciders)

            eval_network = Network()
            bev._candidate_deciders[digest] = SimpleNamespace(network=eval_network)
            with patch("torch.cuda.is_available", return_value=False):
                released = release_candidate_cache(path, expected_checkpoint_sha256=digest)
            self.assertEqual(released["status"], "released")
            self.assertEqual(eval_network.devices, ["cpu"])
            self.assertNotIn(digest, bev._candidate_deciders)


if __name__ == "__main__":
    unittest.main()
