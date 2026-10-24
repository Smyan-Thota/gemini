"""Compact, typed factual state for the sequence operation-facts v3 ablation.

The projection consumes only the current validated public request. It does not
consult labels, a reference implementation, future events, or host files.
"""
from __future__ import annotations

import copy
import json
from collections import Counter
from typing import Any

NATIVE_SCHEMA = "sequence_guard_operation_facts/3"
RULE_TYPES = {"release_limit", "approval_status", "joint_disclosure", "recipient_allowlist"}
LEDGER_NAMES = ("effect-ledger", "authorization-ledger", "artifact-ledger",
                "recipient-ledger", "workflow-ledger")
_STATUS_KEYS = {
    "effect-ledger": "ledger_status",
    "authorization-ledger": "authorization_ledger_status",
    "artifact-ledger": "artifact_ledger_status",
    "recipient-ledger": "recipient_directory_status",
    "workflow-ledger": "ledger_status",
}
_KNOWLEDGE = {"complete", "unavailable", "missing", "malformed", "contradictory", "not_applicable"}
_EFFECT_KEYS = ("path", "destination", "data_units", "data_fields", "grant_ref", "artifact_revision")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_observation(observation: Any) -> tuple[dict[str, Any] | None, bool]:
    if not isinstance(observation, dict):
        return None, True
    if isinstance(observation.get("text"), str):
        try:
            value = json.loads(observation["text"])
        except (TypeError, ValueError):
            return None, True
        if isinstance(value, dict):
            return value, False
        return None, True
    payload = observation.get("payload")
    return (copy.deepcopy(payload), False) if isinstance(payload, dict) else (None, True)


def _event_record(source: str, observation: Any) -> dict[str, Any]:
    if not isinstance(observation, dict):
        return {"source": source, "trust": "untrusted", "text": str(observation)}
    event: dict[str, Any] = {
        "source": observation.get("source", source),
        "trust": observation.get("trust"),
    }
    payload, malformed = _parse_observation(observation)
    if payload is not None:
        event["payload"] = payload
    elif isinstance(observation.get("text"), str):
        # Preserve the original only when it is malformed/non-object JSON.
        event["text"] = observation["text"]
    elif "payload" in observation:
        event["payload"] = copy.deepcopy(observation["payload"])
    elif malformed:
        event["text"] = None
    return event


def _declared_knowledge(rules: dict[str, Any], ledger: str) -> tuple[str, Any, bool]:
    key = _STATUS_KEYS[ledger]
    if key not in rules:
        return "missing", None, False
    status = rules[key]
    if status is None:
        return "malformed", status, True
    if status == "verified":
        return "complete", status, True
    if isinstance(status, str) and status in _KNOWLEDGE:
        return status, status, True
    return "malformed", status, True


def _ledger_events(request: dict[str, Any], source: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for observation in request.get("observations", []):
        if (isinstance(observation, dict) and observation.get("source") == source
                and observation.get("trust") == "trusted"):
            events.append(_event_record(source, observation))
    # Successful upload receipts are nested in the executed action record by
    # the host. Keep them in action order; never infer a receipt from an attempt.
    if source == "effect-ledger":
        for action in request.get("prior_actions", []):
            if not isinstance(action, dict):
                continue
            observation = action.get("observation")
            if (isinstance(observation, dict) and observation.get("source") == source
                    and observation.get("trust") == "trusted"):
                events.append(_event_record(source, observation))
    return events


def _committed_effects(request: dict[str, Any], declared_knowledge: str) -> tuple[dict[str, Any], int, str]:
    prior_actions = request["prior_actions"]
    attempted = sum(1 for action in prior_actions
                    if isinstance(action, dict) and action.get("tool_name") == "upload_file")
    committed: list[dict[str, Any]] = []
    incomplete = False
    for action in prior_actions:
        if not isinstance(action, dict) or action.get("tool_name") != "upload_file":
            continue
        disposition = action.get("host_disposition")
        observation = action.get("observation")
        payload, malformed = _parse_observation(observation)
        if disposition != "ALLOW":
            # A non-ALLOW action is only an attempt. A receipt claiming an
            # effect is inconsistent evidence, so do not count it.
            if payload is not None and payload.get("status") == "committed":
                incomplete = True
            continue
        if (malformed or not isinstance(observation, dict)
                or observation.get("source") != "effect-ledger"
                or observation.get("trust") != "trusted"
                or payload is None or payload.get("status") != "committed"
                or not isinstance(payload.get("effect"), dict)):
            incomplete = True
            continue
        args = action.get("arguments")
        if not isinstance(args, dict):
            incomplete = True
            continue
        expected = {key: args[key] for key in _EFFECT_KEYS if key in args}
        if payload["effect"] != expected:
            incomplete = True
            continue
        committed.append(copy.deepcopy(payload["effect"]))

    if declared_knowledge in {"unavailable", "missing", "malformed", "contradictory"}:
        effect_knowledge = "unknown"
    elif incomplete:
        effect_knowledge = "incomplete"
    else:
        effect_knowledge = "complete"

    units_known = effect_knowledge == "complete" and all(_is_int(item.get("data_units"))
                                                          and item["data_units"] >= 0 for item in committed)
    fields_known = effect_knowledge == "complete" and all(
        isinstance(item.get("data_fields"), list)
        and all(isinstance(value, str) for value in item["data_fields"])
        for item in committed)
    units = sum(item["data_units"] for item in committed) if units_known else None
    fields: list[str] | None = None
    if fields_known:
        seen: set[str] = set()
        fields = []
        for item in committed:
            for value in item["data_fields"]:
                if value not in seen:
                    seen.add(value)
                    fields.append(value)
    history = {
        "attempted_upload_count": attempted,
        "committed_upload_count": len(committed),
        "effect_knowledge": effect_knowledge,
        "committed_data_units": {
            "knowledge": "known" if units_known else "unknown",
            "value": units,
        },
        "committed_data_fields": {
            "knowledge": "known" if fields_known else "unknown",
            "value": fields,
        },
    }
    return history, len(committed), effect_knowledge


def _family_policy_facts(rule_type: str, rules: dict[str, Any]) -> dict[str, Any]:
    # Preserve the exact workflow policy without duplicating its dispatch key.
    # Policy keys remain source facts; no decision is derived here.
    del rule_type
    return {key: copy.deepcopy(value) for key, value in rules.items() if key != "rule_type"}


def encode_operation_facts_v3(validated_request: dict[str, Any]) -> dict[str, Any]:
    """Project one validated ``bev_guard_request/1`` into typed factual fields."""
    request = validated_request
    if not isinstance(request, dict) or request.get("schema") != "bev_guard_request/1":
        raise ValueError("expected validated bev_guard_request/1")
    if not all(key in request for key in ("user_request", "host_policy", "proposal", "prior_actions", "observations")):
        raise ValueError("validated request is missing required public fields")
    policy = request["host_policy"]
    rules = policy.get("workflow_rules") if isinstance(policy, dict) else None
    if not isinstance(rules, dict) or rules.get("rule_type") not in RULE_TYPES:
        raise ValueError("unsupported or missing public workflow rule")
    if not isinstance(policy.get("decision_contract"), str) or not policy["decision_contract"]:
        raise ValueError("public decision contract is missing")
    proposal = request["proposal"]
    if not isinstance(proposal, dict) or not isinstance(proposal.get("arguments"), dict):
        raise ValueError("proposal must contain an arguments object")
    if not isinstance(request["prior_actions"], list) or not isinstance(request["observations"], list):
        raise ValueError("request history and observations must be lists")

    prior_actions = request["prior_actions"]
    actual_allow_count = sum(1 for action in prior_actions
                             if isinstance(action, dict) and action.get("host_disposition") == "ALLOW")
    history, _, effect_knowledge = _committed_effects(request, _declared_knowledge(rules, "effect-ledger")[0])
    history.update({
        "prior_action_record_count": len(prior_actions),
        "actual_allow_count": actual_allow_count,
    })
    bounds = {}
    for key in ("workspace_root", "allowed_operations", "max_operations", "max_write_bytes",
                "operation_semantics", "tool_schemas"):
        if key in policy:
            bounds[key] = copy.deepcopy(policy[key])
    if _is_int(bounds.get("max_operations")):
        bounds["remaining_operations"] = bounds["max_operations"] - actual_allow_count

    arguments = copy.deepcopy(proposal["arguments"])
    output_proposal = {"tool_name": proposal["tool_name"], "arguments": arguments}

    workflow_policy = _family_policy_facts(rules["rule_type"], rules)
    ledgers: dict[str, Any] = {}
    integrity = policy.get("ledger_integrity_anomalies")
    if integrity is None and "ledger_integrity_anomalies" in request:
        integrity = request.get("ledger_integrity_anomalies")
    if integrity is not None:
        ledgers["ledger_integrity_anomalies"] = copy.deepcopy(integrity)
    for source in LEDGER_NAMES:
        knowledge, declared, has_declaration = _declared_knowledge(rules, source)
        events = _ledger_events(request, source)
        malformed_trusted = any(event.get("trust") == "trusted"
                                 and "payload" not in event and "text" in event
                                 and (not isinstance(event.get("text"), str)
                                      or _is_malformed_json(event["text"]))
                                 for event in events)
        if malformed_trusted and knowledge == "complete":
            knowledge = "malformed"
        slot = {"knowledge": knowledge, "events": events}
        if has_declaration:
            slot["declared_status"] = copy.deepcopy(declared)
        ledgers[source] = slot

    observations = copy.deepcopy(request["observations"])
    remaining_top_level = Counter(_canonical_json(item) for item in observations)
    for action in prior_actions:
        if not isinstance(action, dict) or action.get("tool_name") != "read_file":
            continue
        nested = action.get("observation")
        if not isinstance(nested, dict) or set(nested) != {"source", "trust", "text"}:
            continue
        key = _canonical_json(nested)
        if remaining_top_level[key] > 0:
            remaining_top_level[key] -= 1
        else:
            observations.append(copy.deepcopy(nested))

    return {
        "schema": NATIVE_SCHEMA,
        "contract": {"version": "public-decision-semantics-v1", "text": policy["decision_contract"]},
        "task": {"text": request["user_request"]["text"]},
        "bounds": bounds,
        "proposal": output_proposal,
        "history": history,
        "workflow": {"rule_type": rules["rule_type"], "policy_facts": workflow_policy},
        "ledgers": ledgers,
        "observations": observations,
    }


def _is_malformed_json(text: str) -> bool:
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return True
    return not isinstance(value, dict)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)
