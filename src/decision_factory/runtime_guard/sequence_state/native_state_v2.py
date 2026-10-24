"""Native Bev state v2 adds the factual count of previously executed calls."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

if __package__:
    from .native_state_v1 import encode_native_state
else:  # The pinned runtime loader imports these files by their stable basename.
    from native_state_v1 import encode_native_state

NATIVE_SCHEMA = "sequence_guard_native_state/2"


def encode_native_state_v2(request: dict[str, Any]) -> dict[str, Any]:
    """Project a validated public request, including the actual-ALLOW budget count.

    The v1 projection intentionally omitted read receipts from the rollup. That
    made two otherwise-identical histories (one executed read vs one blocked
    read) indistinguishable to the sequence adapter. This version adds only
    the host-disposition count used by the public max_operations contract.
    """
    if not isinstance(request, dict) or request.get("schema") != "bev_guard_request/1":
        raise ValueError("expected bev_guard_request/1")
    prior_actions = request.get("prior_actions")
    if not isinstance(prior_actions, list):
        raise ValueError("prior_actions must be a list")
    actual_allow_count = 0
    for action in prior_actions:
        if not isinstance(action, dict) or action.get("host_disposition") not in {"ALLOW", "BLOCK", "REVIEW"}:
            raise ValueError("prior action has invalid host disposition")
        actual_allow_count += action["host_disposition"] == "ALLOW"

    state = deepcopy(encode_native_state(request))
    state["schema"] = NATIVE_SCHEMA
    state["history"]["actual_allow_count"] = actual_allow_count
    return state
