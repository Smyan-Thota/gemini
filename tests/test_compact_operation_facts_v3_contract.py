"""Frozen fixture integrity checks for the compact operation-facts ablation."""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIXTURES = ROOT / "tests" / "compact_operation_facts_v3_fixtures.jsonl"
BASE_FIXTURES = ROOT / "tests" / "contract_fixtures.jsonl"
CONTRACT = ROOT / "docs" / "COMPACT_OPERATION_FACTS_V3_CONTRACT.md"
EXPECTED_IDS = {
    "complete_empty_not_unknown", "unavailable_not_empty", "missing_not_unavailable",
    "malformed_distinct", "contradictory_distinct", "read_allow_consumes_operation",
    "read_block_does_not_consume", "not_found_read_still_executed",
    "blocked_upload_is_attempt_not_effect", "committed_receipt_counts_once",
    "mismatched_receipt_does_not_match_action", "approval_initial_null_stays_null",
    "wrong_grant_preserved_as_exact_ids", "ordered_auth_revocation_preserved",
    "artifact_last_matching_update_order", "joint_group_null_empty_distinctions",
    "recipient_verified_empty_list", "recipient_unavailable_list",
    "untrusted_observation_retained_not_ledger_event",
    "nested_read_observation_preserved_once",
}
FORBIDDEN_KEYS = {
    "label", "expected", "expected_decision", "verdict", "disposition",
    "safe", "unsafe", "violation", "authorized", "permit", "block", "review",
    "family_id", "group_id", "pair_id", "evaluator", "answer_key",
}


class CompactOperationFactsV3ContractTests(unittest.TestCase):
    def test_frozen_contract_and_20_hand_checkable_fact_boundaries_exist(self) -> None:
        self.assertTrue(CONTRACT.is_file())
        rows = [json.loads(line) for line in FIXTURES.read_text(encoding="utf-8").splitlines() if line]
        base_rows = [json.loads(line) for line in BASE_FIXTURES.read_text(encoding="utf-8").splitlines() if line]
        base_by_id = {row["fixture_id"]: row for row in base_rows}
        base_ids = set(base_by_id)
        self.assertEqual(len(rows), 20)
        self.assertEqual({row["fixture_id"] for row in rows}, EXPECTED_IDS)
        for row in rows:
            self.assertEqual(set(row), {"fixture_id", "base_fixture_id", "patches", "expected_facts"})
            self.assertIn(row["base_fixture_id"], base_ids, row["fixture_id"])
            self.assertIsInstance(row["patches"], list)
            for patch in row["patches"]:
                self.assertEqual(set(patch) - {"value"}, {"op", "path"})
                self.assertIn(patch["op"], {"set", "remove"})
                self.assertTrue(patch["path"].startswith("/"))
                self.assertFalse(FORBIDDEN_KEYS.intersection(_all_keys(patch.get("value"))))
            self.assertTrue(row["expected_facts"], row["fixture_id"])
            self.assertFalse(FORBIDDEN_KEYS.intersection(_all_keys(row["expected_facts"])))
            request = json.loads(json.dumps(base_by_id[row["base_fixture_id"]]["guard_request"]))
            for patch in row["patches"]:
                _apply_patch(request, patch)
            self.assertEqual(request["schema"], "bev_guard_request/1", row["fixture_id"])
            self.assertEqual(set(request), {
                "schema", "user_request", "host_policy", "prior_actions", "observations", "proposal"
            }, row["fixture_id"])

    def test_public_encoder_matches_frozen_fact_assertions(self) -> None:
        from decision_factory.runtime_guard.sequence_state.operation_facts_v3 import (
            encode_operation_facts_v3,
        )
        from decision_factory.runtime_guard.sequence_state.alpha_bind_operation_facts_v3 import (
            restore_operation_facts_v3,
            transform_operation_facts_v3,
        )

        rows = [json.loads(line) for line in FIXTURES.read_text(encoding="utf-8").splitlines() if line]
        base_by_id = {row["fixture_id"]: row for row in (
            json.loads(line) for line in BASE_FIXTURES.read_text(encoding="utf-8").splitlines() if line
        )}
        for row in rows:
            request = json.loads(json.dumps(base_by_id[row["base_fixture_id"]]["guard_request"]))
            for patch in row["patches"]:
                _apply_patch(request, patch)
            state = encode_operation_facts_v3(request)
            self.assertEqual(state["schema"], "sequence_guard_operation_facts/3", row["fixture_id"])
            self.assertEqual(state["contract"]["text"], request["host_policy"]["decision_contract"])
            self.assertFalse(FORBIDDEN_KEYS.intersection(_all_keys(state)), row["fixture_id"])
            self.assertEqual(set(state["proposal"]), {"tool_name", "arguments"}, row["fixture_id"])
            for slot in state["ledgers"].values():
                for event in slot["events"]:
                    self.assertFalse("payload" in event and "text" in event,
                                     f"{row['fixture_id']}: parsed ledger facts serialized twice")
            for path, expected in row["expected_facts"].items():
                try:
                    actual = _get_path(state, path)
                except (KeyError, IndexError, TypeError) as exc:
                    self.fail(f"{row['fixture_id']}: missing fact path {path}: {exc}")
                self.assertEqual(actual, expected, f"{row['fixture_id']}: {path}")
            alpha, sidecar = transform_operation_facts_v3(state)
            self.assertEqual(restore_operation_facts_v3(alpha, sidecar), state, row["fixture_id"])
            self.assertNotIn("source_to_canonical_map", alpha, row["fixture_id"])
            if row["fixture_id"] == "approval_initial_null_stays_null":
                self.assertIsNone(alpha["workflow"]["policy_facts"]["grant"]["approved_artifact_revision"])
                self.assertIsNone(alpha["proposal"]["arguments"]["artifact_revision"])
                alpha_grant = alpha["workflow"]["policy_facts"]["grant"]["grant_ref"]
                alpha_proposal = alpha["proposal"]["arguments"]["grant_ref"]
                self.assertEqual(alpha_grant, alpha_proposal)
            elif row["fixture_id"] == "wrong_grant_preserved_as_exact_ids":
                alpha_grant = alpha["workflow"]["policy_facts"]["grant"]["grant_ref"]
                alpha_proposal = alpha["proposal"]["arguments"]["grant_ref"]
                self.assertNotEqual(alpha_grant, alpha_proposal)
            elif row["fixture_id"] == "joint_group_null_empty_distinctions":
                self.assertIsNone(alpha["workflow"]["policy_facts"]["prohibited_joint_disclosure_groups"])
                self.assertEqual(alpha["proposal"]["arguments"]["data_fields"], [])
            elif row["fixture_id"] == "committed_receipt_counts_once":
                group = alpha["workflow"]["policy_facts"]["prohibited_joint_disclosure_groups"][0]
                committed = alpha["history"]["committed_data_fields"]["value"]
                proposed = alpha["proposal"]["arguments"]["data_fields"]
                self.assertEqual(group[0], committed[0])
                self.assertEqual(group[1], proposed[0])
            elif row["fixture_id"] == "recipient_verified_exact_member":
                approved = alpha["workflow"]["policy_facts"]["approved_recipients"][0]
                destination = alpha["proposal"]["arguments"]["destination"]
                self.assertEqual(approved, destination)


def _all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key.lower()
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def _apply_patch(root, patch):
    path = [part.replace("~1", "/").replace("~0", "~")
            for part in patch["path"].split("/")[1:]]
    parent = root
    for part in path[:-1]:
        parent = parent[int(part)] if isinstance(parent, list) else parent[part]
    key = path[-1]
    if patch["op"] == "remove":
        if isinstance(parent, list):
            del parent[int(key)]
        else:
            del parent[key]
    elif isinstance(parent, list):
        parent[int(key)] = patch["value"]
    else:
        parent[key] = patch["value"]


def _get_path(root, path):
    current = root
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        if part.startswith("["):
            current = current[int(part[1:-1])]
        else:
            current = current[part]
    return current


if __name__ == "__main__":
    unittest.main()
