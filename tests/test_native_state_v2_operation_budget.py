"""Focused regression for executed-operation history in native-state v2."""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_factory.runtime_guard.contract_reference import decide_request  # noqa: E402
from decision_factory.runtime_guard.sequence_state.native_state_v1 import (  # noqa: E402
    encode_native_state,
)
from decision_factory.runtime_guard.sequence_state.native_state_v2 import (  # noqa: E402
    encode_native_state_v2,
)


FIXTURES = ROOT / "tests" / "contract_fixtures.jsonl"


def _approval_request() -> dict:
    for line in FIXTURES.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("fixture_id") == "approval_complete_empty_active":
            return copy.deepcopy(row["guard_request"])
    raise AssertionError("frozen approval fixture missing")


class NativeStateV2OperationBudgetTests(unittest.TestCase):
    def test_prior_read_allow_and_block_have_opposite_budget_labels(self) -> None:
        allow_request = _approval_request()
        allow_request["host_policy"]["max_operations"] = 1
        allow_request["observations"] = [{
            "source": "source-document",
            "trust": "untrusted",
            "text": "The same independently supplied source note is visible in both requests.",
        }]
        allow_request["prior_actions"] = [{
            "tool_name": "read_file",
            "arguments": {"path": "/workspace/source.txt"},
            "host_disposition": "ALLOW",
        }]
        block_request = copy.deepcopy(allow_request)
        block_request["prior_actions"][0]["host_disposition"] = "BLOCK"

        self.assertEqual(
            {key: allow_request[key] for key in allow_request if key != "prior_actions"},
            {key: block_request[key] for key in block_request if key != "prior_actions"},
        )
        self.assertEqual(len(allow_request["prior_actions"]), len(block_request["prior_actions"]))
        self.assertEqual(allow_request["observations"], block_request["observations"])
        self.assertEqual(
            decide_request(allow_request, context="fixed_proposal"),
            {"decision": "BLOCK", "reason": "operation_budget_exhausted"},
        )
        self.assertEqual(
            decide_request(block_request, context="fixed_proposal"),
            {"decision": "PERMIT", "reason": "active_approved_current_revision"},
        )

        # Preserve v1 as the frozen collision baseline while requiring v2 to
        # retain the one fact that changes the reference outcome.
        v1_allow = encode_native_state(allow_request)
        v1_block = encode_native_state(block_request)
        self.assertEqual(v1_allow, v1_block)

        v2_allow = encode_native_state_v2(allow_request)
        v2_block = encode_native_state_v2(block_request)
        self.assertEqual(v2_allow["schema"], "sequence_guard_native_state/2")
        self.assertEqual(v2_block["schema"], "sequence_guard_native_state/2")
        self.assertEqual(v2_allow["history"]["actual_allow_count"], 1)
        self.assertEqual(v2_block["history"]["actual_allow_count"], 0)

        def strip_v2_budget(state: dict) -> dict:
            result = copy.deepcopy(state)
            result["schema"] = "sequence_guard_native_state/1"
            del result["history"]["actual_allow_count"]
            return result

        self.assertEqual(strip_v2_budget(v2_allow), v1_allow)
        self.assertEqual(strip_v2_budget(v2_block), v1_block)


if __name__ == "__main__":
    unittest.main()
