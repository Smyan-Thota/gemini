"""Deterministic, label-free factual summary of visible trusted host ledgers."""
from __future__ import annotations

import copy
import json
from typing import Any

SUMMARY_SOURCE = "host-ledger-summary"
SUMMARY_SCHEMA = "host_ledger_facts/1"
_PROPOSAL_KEYS = ("path", "destination", "data_units", "data_fields", "grant_ref", "artifact_revision")
_POLICY_STATUS_KEYS = (
    "ledger_status", "authorization_ledger_status", "artifact_ledger_status",
    "recipient_directory_status",
)
_SOURCE_SPEC = {
    "authorization-ledger": ("grant_ref", "authorization_ledger_status"),
    "artifact-ledger": ("path", "artifact_ledger_status"),
    "recipient-ledger": ("destination", "recipient_directory_status"),
    "effect-ledger": ("scope", "ledger_status"),
    "workflow-ledger": ("scope", "ledger_status"),
}
_STATUS_MARKERS = {"unavailable", "complete", "verified"}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _iter_observations(request: dict) -> list[tuple[str, dict]]:
    """Return trusted provenance envelopes only; content is parsed by consumers."""
    found: list[tuple[str, dict]] = []
    for observation in request.get("observations", []):
        if isinstance(observation, dict) and isinstance(observation.get("source"), str):
            found.append(("top", observation))
    for action in request.get("prior_actions", []):
        observation = action.get("observation") if isinstance(action, dict) else None
        if isinstance(observation, dict) and isinstance(observation.get("source"), str):
            found.append(("prior", observation))
    return found


def _event_summary(request: dict, source: str, key_field: str, key_value: Any,
                   policy_status: Any, relevant_scopes: list[str] | None = None) -> dict:
    valid_events: list[tuple[int, dict]] = []
    status_payloads: list[dict] = []
    malformed = 0
    for origin, observation in _iter_observations(request):
        if observation.get("source") != source or observation.get("trust") != "trusted":
            continue
        # Prior effect receipts are summarized as receipts, not repeated as ledger events.
        if source == "effect-ledger" and origin == "prior":
            continue
        try:
            payload = json.loads(observation.get("text", ""))
        except (TypeError, ValueError):
            malformed += 1
            continue
        if not isinstance(payload, dict):
            malformed += 1
            continue

        marker = payload.get("status") in _STATUS_MARKERS
        key = payload.get(key_field)
        if key_value is not None:
            if key is None:
                # Unkeyed availability markers apply to the ledger as a whole.
                if not marker:
                    malformed += 1
                    continue
            elif relevant_scopes is not None:
                if key not in relevant_scopes:
                    continue
            elif key != key_value:
                continue
        elif key is not None and relevant_scopes is not None and key not in relevant_scopes:
            continue

        if marker:
            status_payloads.append(payload)
        if "sequence" in payload:
            sequence = payload.get("sequence")
            if not _integer(sequence):
                malformed += 1
                continue
            valid_events.append((sequence, payload))
        elif marker:
            continue
        else:
            # A keyed, trusted event without a usable sequence is retained in the
            # raw request but cannot be ordered in this compact summary.
            malformed += 1

    latest_sequence = max((sequence for sequence, _ in valid_events), default=None)
    latest_events = [] if latest_sequence is None else [
        payload for sequence, payload in valid_events if sequence == latest_sequence
    ]
    dedup: dict[str, dict] = {_canonical(payload): payload for payload in latest_events}
    latest_events = [dedup[key] for key in sorted(dedup)]
    distinct_tie = len(latest_events) > 1

    # Keep ledger availability/status records visible alongside ordered events.
    status_dedup = {_canonical(payload): payload for payload in status_payloads}
    status_payloads = [status_dedup[key] for key in sorted(status_dedup)]
    combined = {_canonical(payload): payload for payload in latest_events + status_payloads}
    latest_payloads = [combined[key] for key in sorted(combined)]

    unavailable = any(payload.get("status") == "unavailable" for payload in status_payloads)
    declared_unavailable = policy_status == "unavailable"
    if distinct_tie:
        state = "conflicting"
    elif latest_events and (malformed or unavailable or declared_unavailable):
        state = "mixed"
    elif latest_events:
        state = "observed"
    elif malformed and (unavailable or declared_unavailable or status_payloads):
        state = "mixed"
    elif malformed:
        state = "malformed"
    elif unavailable or declared_unavailable:
        state = "unavailable"
    elif status_payloads:
        state = "observed"
    else:
        state = "no_matching_record_in_request"

    return {
        "key_field": key_field,
        "key_value": key_value,
        "policy_declared_status": policy_status,
        "state": state,
        "latest_sequence": latest_sequence,
        "latest_payloads": latest_payloads,
        "malformed_matching_record_count": malformed,
    }


def summarize_request(request: dict) -> dict:
    """Build an evidence roll-up without consulting labels, rules' outcomes, or an oracle."""
    if not isinstance(request, dict) or request.get("schema") != "bev_guard_request/1":
        raise ValueError("expected bev_guard_request/1 object")
    proposal = request.get("proposal")
    if not isinstance(proposal, dict) or not isinstance(proposal.get("arguments"), dict):
        raise ValueError("proposal must contain an arguments object")
    args = proposal["arguments"]
    facts_args = {key: copy.deepcopy(args[key]) for key in _PROPOSAL_KEYS if key in args}
    proposal_facts = {"tool_name": proposal.get("tool_name"), "arguments": facts_args}

    prior_actions = request.get("prior_actions", [])
    if not isinstance(prior_actions, list):
        raise ValueError("prior_actions must be a list")
    dispositions = [action.get("host_disposition") if isinstance(action, dict) else None
                    for action in prior_actions]
    non_allow_uploads = sum(
        1 for action in prior_actions
        if isinstance(action, dict) and action.get("tool_name") == "upload_file"
        and action.get("host_disposition") != "ALLOW"
    )

    valid_receipts: list[dict] = []
    missing_or_malformed_allow = 0
    argument_mismatches = 0
    non_allow_committed = 0
    contradiction = False
    units_total = 0
    fields_union: set[str] = set()
    unit_values_complete = True
    field_values_complete = True
    for action in prior_actions:
        if not isinstance(action, dict) or action.get("tool_name") != "upload_file":
            continue
        disposition = action.get("host_disposition")
        observation = action.get("observation")
        trusted_effect = (isinstance(observation, dict)
                          and observation.get("source") == "effect-ledger"
                          and observation.get("trust") == "trusted")
        payload = None
        if trusted_effect:
            try:
                parsed = json.loads(observation.get("text", ""))
                if isinstance(parsed, dict):
                    payload = parsed
            except (TypeError, ValueError):
                payload = None

        is_committed_payload = isinstance(payload, dict) and payload.get("status") == "committed"
        if disposition != "ALLOW":
            if is_committed_payload:
                non_allow_committed += 1
                contradiction = True
            # These attempt records do not contribute to the committed-effect roll-up.
            continue

        if payload is None or payload.get("status") != "committed" or not isinstance(payload.get("effect"), dict):
            missing_or_malformed_allow += 1
            if payload is not None and payload.get("status") == "blocked_no_effect":
                contradiction = True
            continue
        effect = payload["effect"]
        arguments = action.get("arguments")
        if not isinstance(arguments, dict):
            argument_mismatches += 1
            contradiction = True
            continue
        effect_keys = ("path", "destination", "data_units", "data_fields", "grant_ref", "artifact_revision")
        expected_effect = {key: arguments[key] for key in effect_keys if key in arguments}
        if effect != expected_effect:
            argument_mismatches += 1
            contradiction = True
            continue
        if "data_units" in effect and (not _integer(effect["data_units"]) or effect["data_units"] < 0):
            missing_or_malformed_allow += 1
            unit_values_complete = False
            continue
        if "data_fields" in effect and (not isinstance(effect["data_fields"], list)
                                         or any(not isinstance(item, str) for item in effect["data_fields"])):
            missing_or_malformed_allow += 1
            field_values_complete = False
            continue
        valid_receipts.append(effect)
        if "data_units" in effect:
            units_total += effect["data_units"]
        else:
            unit_values_complete = False
        if "data_fields" in effect:
            fields_union.update(effect["data_fields"])
        else:
            field_values_complete = False

    rules = request.get("host_policy", {}).get("workflow_rules", {})
    if not isinstance(rules, dict):
        rules = {}
    policy_statuses = {key: copy.deepcopy(rules[key]) for key in _POLICY_STATUS_KEYS if key in rules}
    declared = {
        "authorization-ledger": policy_statuses.get("authorization_ledger_status"),
        "artifact-ledger": policy_statuses.get("artifact_ledger_status"),
        "recipient-ledger": policy_statuses.get("recipient_directory_status"),
        "effect-ledger": policy_statuses.get("ledger_status"),
        "workflow-ledger": policy_statuses.get("ledger_status"),
    }
    proposal_ref = args.get("grant_ref")
    proposal_path = args.get("path")
    proposal_destination = args.get("destination")
    workflow_scopes = ["older_release_count", "prior_exported_fields"]
    event_specs = {
        "authorization-ledger": ("grant_ref", proposal_ref, None),
        "artifact-ledger": ("path", proposal_path, None),
        "recipient-ledger": ("destination", proposal_destination, None),
        "effect-ledger": ("scope", "committed_effects", None),
        "workflow-ledger": ("scope", workflow_scopes, workflow_scopes),
    }
    latest_events = {}
    for source, (field, target, scopes) in event_specs.items():
        latest_events[source] = _event_summary(
            request, source, field, target, declared[source], relevant_scopes=scopes
        )

    status_markers: dict[str, list[dict]] = {}
    for source, summary in latest_events.items():
        candidates = [payload for payload in summary["latest_payloads"] if "status" in payload]
        if candidates:
            status_markers[source] = [{"source": source, "payload": payload} for payload in candidates]
    markers = [item for source in sorted(status_markers) for item in status_markers[source]]
    effect_unavailable = sum(
        1 for item in status_markers.get("effect-ledger", [])
        if item["payload"].get("status") == "unavailable"
    )
    effect_state = latest_events["effect-ledger"]["state"]
    workflow_marker_payloads = [item["payload"] for item in status_markers.get("workflow-ledger", [])]
    old_units_unavailable = any(p.get("status") == "unavailable" and p.get("scope") == "older_release_count"
                                for p in workflow_marker_payloads)
    old_fields_unavailable = any(p.get("status") == "unavailable" and p.get("scope") == "prior_exported_fields"
                                 for p in workflow_marker_payloads)
    ledger_status_unavailable = policy_statuses.get("ledger_status") == "unavailable"
    if ledger_status_unavailable or effect_unavailable or old_units_unavailable:
        unit_values_complete = False
    if ledger_status_unavailable or effect_unavailable or old_fields_unavailable:
        field_values_complete = False

    receipt_state = "complete"
    if contradiction:
        receipt_state = "contradictory"
    elif (missing_or_malformed_allow or effect_unavailable or effect_state in {"malformed", "mixed", "conflicting"}
          or old_units_unavailable or old_fields_unavailable or ledger_status_unavailable):
        receipt_state = "incomplete"
    if receipt_state != "complete":
        unit_values_complete = False
        field_values_complete = False

    return {
        "schema": SUMMARY_SCHEMA,
        "proposal_facts": proposal_facts,
        "prior_action_facts": {
            "prior_action_record_count": len(prior_actions),
            "non_allow_upload_record_count": non_allow_uploads,
            "disposition_alone_is_receipt_proof": False,
            "non_allow_uploads_added_to_committed_totals": False,
        },
        "committed_effect_facts": {
            "valid_committed_receipt_count": len(valid_receipts),
            "known_committed_units_subtotal": units_total,
            "known_committed_field_union": sorted(fields_union),
            "units_complete": bool(unit_values_complete),
            "fields_complete": bool(field_values_complete),
            "receipt_evidence_state": receipt_state,
            "missing_or_malformed_allow_upload_receipt_count": missing_or_malformed_allow,
            "argument_mismatch_receipt_count": argument_mismatches,
            "non_allow_committed_receipt_count": non_allow_committed,
            "unavailable_effect_ledger_marker_count": effect_unavailable,
        },
        "policy_declared_ledger_statuses": policy_statuses,
        "trusted_status_markers": markers,
        "latest_relevant_events": latest_events,
        "summary_note": (
            "Host-derived facts from trusted records visible in this request. Recorded totals use only "
            "matching committed receipts; this summary does not classify the proposed action."
        ),
    }


def append_host_ledger_summary(request: dict) -> dict:
    """Return a copy with one trusted, deterministic factual-summary observation appended."""
    if any(isinstance(item, dict) and item.get("source") == SUMMARY_SOURCE
           for item in request.get("observations", [])):
        raise ValueError("request already contains a host-ledger-summary observation")
    result = copy.deepcopy(request)
    if not isinstance(result.get("observations"), list):
        raise ValueError("observations must be a list")
    summary = summarize_request(result)
    result["observations"].append({
        "source": SUMMARY_SOURCE,
        "trust": "trusted",
        "text": _canonical(summary),
    })
    return result
