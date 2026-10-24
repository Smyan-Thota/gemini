"""Policy-agnostic bijective renaming for opaque entity identifiers."""
from __future__ import annotations

import copy
import re
from typing import Any

SCHEMA = "sequence_guard_native_state_alpha/1"
_IDENTIFIER_FIELDS = ("destination", "grant_ref", "artifact_revision",
                      "approved_artifact_revision", "initial_artifact_revision",
                      "revision")


def _list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        raise ValueError("expected a list of opaque string identifiers")
    return value


def _collect(state: dict[str, Any]) -> list[str]:
    values: list[str] = []
    policy = state.get("policy", {})
    rules = policy.get("workflow_rules", {}) if isinstance(policy, dict) else {}
    if isinstance(rules, dict):
        if "approved_recipients" in rules:
            values.extend(_list_of_strings(rules["approved_recipients"]))
        grant = rules.get("grant")
        if isinstance(grant, dict):
            for key in ("grant_ref", "approved_artifact_revision", "initial_artifact_revision"):
                if isinstance(grant.get(key), str): values.append(grant[key])
        if "prohibited_joint_disclosure_groups" in rules:
            groups = rules["prohibited_joint_disclosure_groups"]
            if not isinstance(groups, list): raise ValueError("invalid disclosure groups")
            for group in groups: values.extend(_list_of_strings(group))
    proposal = state.get("proposal", {})
    args = proposal.get("arguments", {}) if isinstance(proposal, dict) else {}
    if isinstance(args, dict):
        for key in ("destination", "grant_ref", "artifact_revision"):
            if isinstance(args.get(key), str): values.append(args[key])
        if "data_fields" in args: values.extend(_list_of_strings(args["data_fields"]))
    history = state.get("history", {})
    if isinstance(history, dict):
        committed = history.get("committed_effects", {})
        projected = history.get("projected_effect_facts", {})
        if isinstance(committed, dict) and "known_field_union" in committed:
            values.extend(_list_of_strings(committed["known_field_union"]))
        if isinstance(projected, dict) and "projected_field_union" in projected:
            values.extend(_list_of_strings(projected["projected_field_union"]))
    events = state.get("matching_event_history", [])
    if not isinstance(events, list): raise ValueError("invalid event history")
    for event in events:
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if isinstance(payload, dict):
            for key in _IDENTIFIER_FIELDS:
                if isinstance(payload.get(key), str): values.append(payload[key])
            if "data_fields" in payload: values.extend(_list_of_strings(payload["data_fields"]))
    # Deterministic first-seen order, retaining one token per exact string.
    return list(dict.fromkeys(values))


def _replace_text(text: str, mapping: dict[str, str]) -> str:
    # Treat identifier punctuation as part of the token so `rev-1` cannot
    # partially rewrite `rev-10`. This only touches exact references in prose.
    for source in sorted(mapping, key=lambda x: (-len(x), x)):
        # Dot is part of opaque hostnames, but a final prose period after a
        # hostname is punctuation. Reject dots that continue another token.
        pattern = rf"(?<![A-Za-z0-9_.:/-]){re.escape(source)}(?![A-Za-z0-9_:/-]|\.(?=[A-Za-z0-9_:/-]))"
        text = re.sub(pattern, mapping[source], text)
    return text


def _map_fields(state: dict[str, Any], mapping: dict[str, str], *, reverse: bool = False) -> None:
    if reverse: mapping = {v: k for k, v in mapping.items()}
    def rename(value: Any) -> Any:
        if not isinstance(value, str): raise ValueError("identifier values must be strings")
        return mapping.get(value, value)
    policy = state.get("policy", {})
    rules = policy.get("workflow_rules", {}) if isinstance(policy, dict) else {}
    if isinstance(rules, dict):
        if isinstance(rules.get("approved_recipients"), list):
            rules["approved_recipients"] = [rename(x) for x in rules["approved_recipients"]]
        grant = rules.get("grant")
        if isinstance(grant, dict):
            for key in ("grant_ref", "approved_artifact_revision", "initial_artifact_revision"):
                if key in grant: grant[key] = rename(grant[key])
        if isinstance(rules.get("prohibited_joint_disclosure_groups"), list):
            rules["prohibited_joint_disclosure_groups"] = [
                [rename(x) for x in group] for group in rules["prohibited_joint_disclosure_groups"]]
    proposal = state.get("proposal", {})
    args = proposal.get("arguments", {}) if isinstance(proposal, dict) else {}
    if isinstance(args, dict):
        for key in ("destination", "grant_ref", "artifact_revision"):
            if key in args: args[key] = rename(args[key])
        if isinstance(args.get("data_fields"), list): args["data_fields"] = [rename(x) for x in args["data_fields"]]
    history = state.get("history", {})
    if isinstance(history, dict):
        committed = history.get("committed_effects", {})
        projected = history.get("projected_effect_facts", {})
        if isinstance(committed, dict) and isinstance(committed.get("known_field_union"), list):
            committed["known_field_union"] = [rename(x) for x in committed["known_field_union"]]
        if isinstance(projected, dict) and isinstance(projected.get("projected_field_union"), list):
            projected["projected_field_union"] = [rename(x) for x in projected["projected_field_union"]]
    for event in state.get("matching_event_history", []):
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if isinstance(payload, dict):
            for key in _IDENTIFIER_FIELDS:
                if key in payload: payload[key] = rename(payload[key])
            if isinstance(payload.get("data_fields"), list): payload["data_fields"] = [rename(x) for x in payload["data_fields"]]
    task = state.get("task", {})
    if isinstance(task, dict) and isinstance(task.get("text"), str):
        task["text"] = _replace_text(task["text"], mapping)
    for obs in state.get("context_observations", []):
        if isinstance(obs, dict) and isinstance(obs.get("text"), str):
            obs["text"] = _replace_text(obs["text"], mapping)


def transform_model_state(state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Return renamed model state and reversible map; no labels/verdicts accepted."""
    if not isinstance(state, dict) or state.get("schema") != "sequence_guard_native_state/1":
        raise ValueError("expected sequence_guard_native_state/1")
    copied = copy.deepcopy(state)
    ordered = _collect(copied)
    mapping = {value: f"ENTITY_{i:03d}" for i, value in enumerate(ordered, 1)}
    _map_fields(copied, mapping)
    copied["schema"] = SCHEMA
    return copied, mapping


def restore_model_state(state: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """Inverse used for CPU invariants/audit; never sent to the model."""
    if not isinstance(state, dict) or state.get("schema") != SCHEMA:
        raise ValueError("expected alpha-bound native state")
    restored = copy.deepcopy(state)
    restored["schema"] = "sequence_guard_native_state/1"
    _map_fields(restored, mapping, reverse=True)
    return restored
