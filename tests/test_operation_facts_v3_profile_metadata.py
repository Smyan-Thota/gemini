"""CPU-only metadata binding checks for the V3 input profile and V9 warm prior."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_factory.runtime_guard import bev, sequence_candidate, sequence_v2, sequence_v3  # noqa: E402


class OperationFactsV3ProfileMetadataTests(unittest.TestCase):
    def test_candidate_profile_pins_and_unpromoted_v9_cross_profile_are_explicit(self) -> None:
        pins = sequence_v3.source_pins()
        self.assertEqual(pins["contract_sha256"], sequence_candidate._V3_CONTRACT_SHA256)
        v3_payload = {
            "schema": "bev_lora_head_checkpoint/2",
            "base_model_id": bev.MODEL_ID,
            "base_revision": bev.MODEL_REVISION,
            "base_weight_sha256": bev.MODEL_SHA256,
            "training_input_mode": sequence_candidate._V3_TRAINING_INPUT_MODE,
            "candidate_training_contract": sequence_candidate._V3_TRAINING_CONTRACT,
            "model_state_schema": "sequence_guard_operation_facts_alpha/3",
            "model_state_encoder_sha256": pins["operation_facts_v3_sha256"],
            "model_state_transform_sha256": pins["alpha_bind_operation_facts_v3_sha256"],
            "model_state_alpha_base_sha256": pins["alpha_bind_v1_sha256"],
            "model_state_contract_sha256": pins["contract_sha256"],
            "model_state_manifest_sha256": "a" * 64,
            "model_state_independent_audit_sha256": "b" * 64,
            "selected_step": 16,
        }
        _, role, eligible = sequence_candidate._validate_checkpoint_metadata(
            v3_payload,
            checkpoint_sha256="c" * 64,
            allow_unpromoted_v9_prior=False,
            input_profile="operation_facts_v3",
        )
        self.assertEqual(role, "candidate")
        self.assertTrue(eligible)

        v9_prior = {
            "schema": "bev_lora_head_checkpoint/2",
            "base_model_id": bev.MODEL_ID,
            "base_revision": bev.MODEL_REVISION,
            "base_weight_sha256": bev.MODEL_SHA256,
            "training_input_mode": "stored_alpha_v2_state_no_runtime_retransform",
            "candidate_training_contract": sequence_candidate._V9_PRIOR_CONTRACT,
            "model_state_schema": "sequence_guard_native_state_alpha/2",
            "model_state_manifest_sha256": sequence_candidate._V9_PRIOR_MANIFEST_SHA256,
            "model_state_independent_audit_sha256": None,
            "model_state_transform_sha256": sequence_v2.ALPHA_BIND_V2_SHA256,
            "selected_step": 64,
        }
        returned, prior_role, prior_eligible = sequence_candidate._validate_checkpoint_metadata(
            v9_prior,
            checkpoint_sha256=sequence_candidate._V9_PRIOR_SHA256,
            allow_unpromoted_v9_prior=True,
            input_profile="operation_facts_v3",
        )
        self.assertEqual(returned["training_input_mode"], "stored_alpha_v2_state_no_runtime_retransform")
        self.assertEqual(returned["model_state_schema"], "sequence_guard_native_state_alpha/2")
        self.assertEqual(prior_role, "unpromoted_warm_start_prior")
        self.assertFalse(prior_eligible)


if __name__ == "__main__":
    unittest.main()
