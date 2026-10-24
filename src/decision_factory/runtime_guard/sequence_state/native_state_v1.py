"""Native Bev encoder state derived from audited sequence ledger facts."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compact_state_v3 import compact_model_request  # noqa: E402

NATIVE_SCHEMA = "sequence_guard_native_state/1"
RELEVANT_LEDGER_SOURCES = {
    "release_limit": ("effect-ledger", "workflow-ledger"),
    "approval_status": ("authorization-ledger", "artifact-ledger"),
    "joint_disclosure": ("effect-ledger", "workflow-ledger"),
    "recipient_allowlist": ("recipient-ledger",),
}


def encode_native_state(request: dict[str, Any]) -> dict[str, Any]:
    """Project a validated public request into direct structured model fields.

    The caller retains `request` separately as host-side evidence. Labels and
    evaluator state are not accepted by this function.
    """
    if not isinstance(request, dict) or request.get("schema") != "bev_guard_request/1":
        raise ValueError("expected bev_guard_request/1")
    policy = request.get("host_policy")
    rules = policy.get("workflow_rules") if isinstance(policy, dict) else None
    if not isinstance(rules, dict) or rules.get("rule_type") not in RELEVANT_LEDGER_SOURCES:
        raise ValueError("unsupported or missing public workflow rule")
    proposal = request.get("proposal")
    if not isinstance(proposal, dict) or not isinstance(proposal.get("arguments"), dict):
        raise ValueError("proposal must include exact arguments")

    compact_request = compact_model_request(request)
    summary_obs = next(obs for obs in compact_request["observations"]
                       if isinstance(obs, dict) and obs.get("source") == "host-ledger-summary")
    facts = json.loads(summary_obs["text"])
    rule_type = rules["rule_type"]
    action_tool = proposal["tool_name"]
    # Preserve complete public policy/tool schemas so this projection is a
    # format change, not a policy restriction or action allowlist experiment.
    policy_facts = copy.deepcopy(policy)

    original_observations = request.get("observations", [])
    if not isinstance(original_observations, list):
        raise ValueError("observations must be a list")
    context = [
        copy.deepcopy(obs) for obs in original_observations
        if isinstance(obs, dict) and not (
            obs.get("trust") == "trusted"
            and obs.get("source") in {
                "authorization-ledger", "artifact-ledger", "recipient-ledger",
                "effect-ledger", "workflow-ledger", "host-ledger-summary",
            }
        )
    ]
    relevant = set(RELEVANT_LEDGER_SOURCES[rule_type])
    ledger_states = {source: copy.deepcopy(facts["ledger_states"][source])
                     for source in relevant if source in facts["ledger_states"]}
    # Global integrity anomalies remain visible because the reference treats
    # malformed or unordered trusted event streams as request-wide evidence gaps.
    ledger_integrity = copy.deepcopy(facts["ledger_integrity_anomalies"])
    matching_events = copy.deepcopy(facts["matching_event_history"])

    return {
        "schema": NATIVE_SCHEMA,
        "task": {"text": request["user_request"]["text"], "trust": "trusted"},
        "policy": policy_facts,
        "proposal": copy.deepcopy(proposal),
        "history": {
            "prior_action_record_count": facts["history"]["prior_action_record_count"],
            "non_allow_upload_record_count": facts["history"]["non_allow_upload_record_count"],
            "committed_effects": copy.deepcopy(facts["history"]["committed_effects"]),
            "projected_effect_facts": copy.deepcopy(facts["proposal_arithmetic_facts"]),
        },
        "ledger_state": ledger_states,
        "ledger_integrity_anomalies": ledger_integrity,
        "matching_event_history": matching_events,
        "context_observations": context,
    }
