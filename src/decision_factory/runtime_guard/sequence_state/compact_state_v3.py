"""Compact, auditable model projection for the four frozen sequence workflow rules.

The source request must remain outside this projection as immutable evidence. This
module emits only a model-facing bev_guard_request/1 object and never reads labels.
"""
from __future__ import annotations

import copy
import json
from typing import Any

from pathlib import Path
import sys
V2_DIR = Path(__file__).resolve().parents[1] / "summary-v2"
sys.path.insert(0, str(V2_DIR))
from ledger_summary import summarize_request  # noqa: E402

COMPACT_SCHEMA = "host_ledger_facts/2"
LEDGER_SOURCES = {
    "authorization-ledger": ("grant_ref",),
    "artifact-ledger": ("path",),
    "recipient-ledger": ("destination",),
    "workflow-ledger": ("scope",),
    "effect-ledger": ("scope",),
}
EVENTS_TO_KEEP = {
    "authorization-ledger", "artifact-ledger", "recipient-ledger",
    "workflow-ledger", "effect-ledger",
}
RULE_TYPES = {"release_limit", "approval_status", "joint_disclosure", "recipient_allowlist"}
PROPOSAL_MATCH = {
    "authorization-ledger": ("grant_ref", "grant_ref"),
    "artifact-ledger": ("path", "path"),
    "recipient-ledger": ("destination", "destination"),
    "effect-ledger": ("scope", None),
    "workflow-ledger": ("scope", None),
}
WORKFLOW_SCOPES = {"older_release_count", "prior_exported_fields"}


def _payload(observation: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    try:
        value = json.loads(observation.get("text", ""))
    except (TypeError, ValueError):
        return None, True
    return (value, False) if isinstance(value, dict) else (None, True)


def _is_relevant(source: str, payload: dict[str, Any], request: dict[str, Any]) -> bool:
    field, arg_name = PROPOSAL_MATCH[source]
    if source == "effect-ledger":
        target = "committed_effects"
    elif source == "workflow-ledger":
        target = None
    else:
        target = request["proposal"]["arguments"].get(arg_name)
    marker = payload.get("status") in {"unavailable", "complete", "verified"}
    value = payload.get(field)
    if value is None:
        return marker
    if source == "workflow-ledger":
        return value in WORKFLOW_SCOPES
    if target is None:
        return marker
    return value == target


def _matching_event_history(request: dict[str, Any]) -> list[dict[str, Any]]:
    """Preserve every proposal-matching trusted event; never collapse timeline to latest."""
    result = []
    for observation in request.get("observations", []):
        if not isinstance(observation, dict):
            continue
        source = observation.get("source")
        if source not in EVENTS_TO_KEEP or observation.get("trust") != "trusted":
            continue
        payload, malformed = _payload(observation)
        if malformed:
            result.append({"source": source, "trust": "trusted", "malformed": True})
        elif _is_relevant(source, payload, request):
            result.append({"source": source, "trust": "trusted", "payload": payload})
    return result


def _event_integrity_facts(request: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Record source-wide shape/order facts used by the frozen history checker."""
    facts = {}
    for source in sorted(EVENTS_TO_KEEP):
        records = [obs for obs in request.get("observations", [])
                   if isinstance(obs, dict) and obs.get("source") == source
                   and obs.get("trust") == "trusted"]
        sequences = []
        malformed = invalid_sequence = unavailable = 0
        for obs in records:
            payload, bad = _payload(obs)
            if bad:
                malformed += 1
                continue
            if payload.get("status") == "unavailable":
                unavailable += 1
                continue
            seq = payload.get("sequence")
            if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
                invalid_sequence += 1
            else:
                sequences.append(seq)
        seen = set()
        duplicates = 0
        out_of_order = 0
        previous = None
        for seq in sequences:
            if seq in seen:
                duplicates += 1
            if previous is not None and seq < previous:
                out_of_order += 1
            seen.add(seq)
            previous = seq
        facts[source] = {
            "trusted_record_count": len(records),
            "malformed_payload_count": malformed,
            "invalid_or_missing_sequence_count": invalid_sequence,
            "duplicate_sequence_count": duplicates,
            "out_of_order_sequence_count": out_of_order,
            "unavailable_marker_count": unavailable,
        }
    return facts


def compact_model_request(request: dict[str, Any]) -> dict[str, Any]:
    """Preserve public task/policy/proposal and project receipt/event history to facts."""
    if not isinstance(request, dict) or request.get("schema") != "bev_guard_request/1":
        raise ValueError("expected bev_guard_request/1")
    policy = request.get("host_policy")
    rules = policy.get("workflow_rules") if isinstance(policy, dict) else None
    if not isinstance(rules, dict) or rules.get("rule_type") not in RULE_TYPES:
        raise ValueError("unknown workflow rule; route through legacy/review path")

    base = summarize_request(request)
    source_integrity = _event_integrity_facts(request)
    ledger_index = {}
    integrity_index = {}
    for source in sorted(EVENTS_TO_KEEP):
        event = base["latest_relevant_events"][source]
        integrity = source_integrity[source]
        state_record = {"state": event["state"]}
        if event["latest_sequence"] is not None:
            state_record["latest_sequence"] = event["latest_sequence"]
        if event["malformed_matching_record_count"]:
            state_record["malformed_matching_record_count"] = event["malformed_matching_record_count"]
        ledger_index[source] = state_record
        anomalies = {key: value for key, value in integrity.items()
                     if key != "trusted_record_count" and value}
        if anomalies:
            anomalies["trusted_record_count"] = integrity["trusted_record_count"]
            integrity_index[source] = anomalies
    args = request["proposal"]["arguments"]
    effects = base["committed_effect_facts"]
    proposed_units = args.get("data_units")
    if isinstance(proposed_units, int) and not isinstance(proposed_units, bool):
        effects_projection = {
            "known_used_units_subtotal": effects["known_committed_units_subtotal"],
            "proposed_release_units": proposed_units,
            "projected_units_subtotal": effects["known_committed_units_subtotal"] + proposed_units,
            "based_on_known_receipts_only": True,
            "history_complete": effects["units_complete"],
        }
    else:
        effects_projection = {"proposed_release_units": None, "history_complete": effects["units_complete"]}
    prior_fields = set(effects["known_committed_field_union"])
    proposed_fields = args.get("data_fields")
    if isinstance(proposed_fields, list) and all(isinstance(x, str) for x in proposed_fields):
        effects_projection["projected_field_union"] = sorted(prior_fields | set(proposed_fields))
        effects_projection["field_history_complete"] = effects["fields_complete"]
    else:
        effects_projection["projected_field_union"] = None
        effects_projection["field_history_complete"] = effects["fields_complete"]
    facts = {
        "schema": COMPACT_SCHEMA,
        "history": {
            "prior_action_record_count": base["prior_action_facts"]["prior_action_record_count"],
            "non_allow_upload_record_count": base["prior_action_facts"]["non_allow_upload_record_count"],
            "committed_effects": {
                "valid_receipt_count": effects["valid_committed_receipt_count"],
                "known_units_subtotal": effects["known_committed_units_subtotal"],
                "known_field_union": effects["known_committed_field_union"],
                "units_complete": effects["units_complete"],
                "fields_complete": effects["fields_complete"],
                "evidence_state": effects["receipt_evidence_state"],
                "missing_or_malformed_allow_receipt_count": effects["missing_or_malformed_allow_upload_receipt_count"],
                "argument_mismatch_receipt_count": effects["argument_mismatch_receipt_count"],
                "non_allow_committed_receipt_count": effects["non_allow_committed_receipt_count"],
                "unavailable_effect_ledger_marker_count": effects["unavailable_effect_ledger_marker_count"],
            },
        },
        "ledger_states": ledger_index,
        "ledger_integrity_anomalies": integrity_index,
        "matching_event_history": _matching_event_history(request),
        "proposal_arithmetic_facts": effects_projection,
        "summary_note": "Factual ledger rollup; source, policy, task, proposal, and untrusted observations remain in the request. Integrity counters are omitted when zero.",
    }
    compact = copy.deepcopy(request)
    compact["prior_actions"] = []
    original_observations = request.get("observations", [])
    if not isinstance(original_observations, list):
        raise ValueError("observations must be a list")
    # Keep untrusted observations byte-for-value and all trusted non-ledger context.
    # Trusted receipt/event observations are represented in facts above.
    compact["observations"] = [
        copy.deepcopy(obs) for obs in original_observations
        if not (isinstance(obs, dict) and obs.get("trust") == "trusted"
                and obs.get("source") in LEDGER_SOURCES)
    ]
    compact["observations"].append({
        "source": "host-ledger-summary",
        "trust": "trusted",
        "text": json.dumps(facts, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False),
    })
    return compact
