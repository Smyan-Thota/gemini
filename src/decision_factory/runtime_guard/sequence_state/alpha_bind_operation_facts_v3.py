"""Alpha-binding adapter for compact operation facts v3.

It delegates deterministic ID ordering/text tokenization to the frozen v1
bijection, then applies that exact map to the v3 field layout. Sidecar maps are
for audit/reconstruction only and must never be serialized into model input.
"""
from __future__ import annotations

import copy
from typing import Any

from . import alpha_bind_v1
from .operation_facts_v3 import NATIVE_SCHEMA

SCHEMA = "sequence_guard_operation_facts_alpha/3"
_V1_SCHEMA = "sequence_guard_native_state_alpha/1"
_ID_FIELDS = {"destination", "grant_ref", "artifact_revision",
              "approved_artifact_revision", "initial_artifact_revision", "revision"}
_ID_LISTS = {"approved_recipients", "data_fields", "known_field_union",
             "prohibited_joint_disclosure_groups", "committed_data_fields"}


def _clean_list(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    return [item for item in value if isinstance(item, str)]


def _seed_event(event: dict[str, Any]) -> dict[str, Any]:
    result = {key: copy.deepcopy(event[key]) for key in ("source", "trust") if key in event}
    payload = event.get("payload")
    if isinstance(payload, dict):
        clean_payload = {}
        for key, value in payload.items():
            if key in _ID_FIELDS:
                if isinstance(value, str):
                    clean_payload[key] = value
            elif key == "data_fields":
                fields = _clean_list(value)
                if fields is not None:
                    clean_payload[key] = fields
            else:
                clean_payload[key] = copy.deepcopy(value)
        result["payload"] = clean_payload
    if isinstance(event.get("text"), str):
        result["text"] = event["text"]
    return result


def _seed_for_v1(state: dict[str, Any]) -> dict[str, Any]:
    """Place v3 identifiers in the frozen v1 traversal order for identical IDs."""
    workflow = state.get("workflow", {}).get("policy_facts", {})
    if not isinstance(workflow, dict):
        workflow = {}
    rules: dict[str, Any] = {}
    for key in ("approved_recipients",):
        value = _clean_list(workflow.get(key))
        if value is not None:
            rules[key] = value
    grant = workflow.get("grant")
    if isinstance(grant, dict):
        rules["grant"] = {key: grant[key] for key in (
            "grant_ref", "approved_artifact_revision", "initial_artifact_revision")
            if isinstance(grant.get(key), str)}
    groups = workflow.get("prohibited_joint_disclosure_groups")
    if isinstance(groups, list):
        rules["prohibited_joint_disclosure_groups"] = [
            [value for value in group if isinstance(value, str)] if isinstance(group, list) else []
            for group in groups]

    proposal = state.get("proposal", {})
    args = proposal.get("arguments", {}) if isinstance(proposal, dict) else {}
    seed_args: dict[str, Any] = {}
    if isinstance(args, dict):
        for key in ("destination", "grant_ref", "artifact_revision"):
            if isinstance(args.get(key), str):
                seed_args[key] = args[key]
        fields = _clean_list(args.get("data_fields"))
        if fields is not None:
            seed_args["data_fields"] = fields

    history = state.get("history", {})
    fields_fact = history.get("committed_data_fields", {}) if isinstance(history, dict) else {}
    known_fields = _clean_list(fields_fact.get("value")) if isinstance(fields_fact, dict) else None
    committed = {"known_field_union": known_fields} if known_fields is not None else {}
    event_rows = []
    ledgers = state.get("ledgers", {})
    if isinstance(ledgers, dict):
        for ledger_name in ("effect-ledger", "authorization-ledger", "artifact-ledger",
                            "recipient-ledger", "workflow-ledger"):
            slot = ledgers.get(ledger_name, {})
            if isinstance(slot, dict):
                for event in slot.get("events", []):
                    if isinstance(event, dict):
                        event_rows.append(_seed_event(event))
    return {
        "schema": "sequence_guard_native_state/1",
        "policy": {"workflow_rules": rules},
        "proposal": {"arguments": seed_args},
        "history": {"committed_effects": committed, "projected_effect_facts": {}},
        "matching_event_history": event_rows,
        "task": copy.deepcopy(state.get("task", {})),
        "context_observations": copy.deepcopy(state.get("observations", [])),
    }


def _rename_list(value: list[Any], mapping: dict[str, str], *, nested: bool = False) -> list[Any]:
    result = []
    for item in value:
        if isinstance(item, str):
            result.append(mapping.get(item, item))
        elif nested and isinstance(item, list):
            result.append(_rename_list(item, mapping))
        else:
            result.append(copy.deepcopy(item))
    return result


def _apply_mapping(value: Any, mapping: dict[str, str], reverse: bool) -> Any:
    if reverse:
        mapping = {canonical: source for source, canonical in mapping.items()}

    def walk(item: Any, key: str | None = None, parent: str | None = None) -> Any:
        if isinstance(item, dict):
            return {k: walk(v, k, key) for k, v in item.items()}
        if isinstance(item, list):
            if key in _ID_LISTS or (key == "value" and parent == "committed_data_fields"):
                return _rename_list(item, mapping, nested=(key == "prohibited_joint_disclosure_groups"))
            return [walk(v, None, key) for v in item]
        if isinstance(item, str):
            if key in _ID_FIELDS:
                return mapping.get(item, item)
            if key == "text":
                return alpha_bind_v1._replace_text(item, mapping)
        return copy.deepcopy(item)

    return walk(value)


def transform_operation_facts_v3(state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    if not isinstance(state, dict) or state.get("schema") != NATIVE_SCHEMA:
        raise ValueError(f"expected {NATIVE_SCHEMA}")
    seed = _seed_for_v1(state)
    _unused_transformed, mapping = alpha_bind_v1.transform_model_state(seed)
    transformed = _apply_mapping(state, mapping, reverse=False)
    transformed["schema"] = SCHEMA
    return transformed, mapping


def restore_operation_facts_v3(state: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    if not isinstance(state, dict) or state.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA}")
    restored = _apply_mapping(state, mapping, reverse=True)
    restored["schema"] = NATIVE_SCHEMA
    return restored
