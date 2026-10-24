"""Alpha binder v2 preserves v1 identifier mapping and native-state v2 facts."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if __package__:
    from . import alpha_bind_v1  # noqa: E402
else:  # The pinned runtime loader imports these files by their stable basename.
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import alpha_bind_v1  # noqa: E402

NATIVE_SCHEMA = "sequence_guard_native_state/2"
SCHEMA = "sequence_guard_native_state_alpha/2"
V1_NATIVE_SCHEMA = "sequence_guard_native_state/1"
V1_ALPHA_SCHEMA = "sequence_guard_native_state_alpha/1"


def _strip_nullable_identifiers(source: dict[str, Any]) -> tuple[list, list]:
    null_fields: list[tuple[tuple[Any, ...], str]] = []
    null_list_items: list[tuple[tuple[Any, ...], list[int]]] = []

    def omit_nullable(mapping: Any, path: tuple[Any, ...], keys: tuple[str, ...]) -> None:
        if not isinstance(mapping, dict):
            return
        for key in keys:
            if key in mapping and mapping[key] is None:
                null_fields.append((path, key))
                del mapping[key]
            elif key in mapping and isinstance(mapping[key], list):
                values = mapping[key]
                indices = [index for index, value in enumerate(values) if value is None]
                if indices:
                    null_list_items.append((path + (key,), indices))
                    mapping[key] = [value for value in values if value is not None]

    rules_path = ("policy", "workflow_rules")

    policy = source.get("policy", {})
    rules = policy.get("workflow_rules", {}) if isinstance(policy, dict) else {}
    omit_nullable(rules, rules_path, ("approved_recipients",))
    if isinstance(rules, dict) and rules.get("prohibited_joint_disclosure_groups", object()) is None:
        null_fields.append((rules_path, "prohibited_joint_disclosure_groups"))
        del rules["prohibited_joint_disclosure_groups"]
    if isinstance(rules, dict) and isinstance(rules.get("prohibited_joint_disclosure_groups"), list):
        original_groups = rules["prohibited_joint_disclosure_groups"]
        # Keep null groups out of v1's string-list validator, then restore them
        # before restoring any null member inside an individual group.
        for index, group in enumerate(original_groups):
            if isinstance(group, list):
                null_indices = [i for i, value in enumerate(group) if value is None]
                if null_indices:
                    null_list_items.append((rules_path + ("prohibited_joint_disclosure_groups", index),
                                            null_indices))
                    original_groups[index] = [value for value in group if value is not None]
        groups = rules["prohibited_joint_disclosure_groups"]
        group_indices = [index for index, group in enumerate(groups) if group is None]
        if group_indices:
            null_list_items.append((rules_path + ("prohibited_joint_disclosure_groups",), group_indices))
            rules["prohibited_joint_disclosure_groups"] = [g for g in groups if g is not None]
    grant = rules.get("grant") if isinstance(rules, dict) else None
    omit_nullable(grant, rules_path + ("grant",),
                  ("grant_ref", "approved_artifact_revision", "initial_artifact_revision"))

    proposal = source.get("proposal", {})
    args = proposal.get("arguments", {}) if isinstance(proposal, dict) else {}
    omit_nullable(args, ("proposal", "arguments"),
                  ("destination", "grant_ref", "artifact_revision", "data_fields"))
    history = source.get("history", {})
    committed = history.get("committed_effects", {}) if isinstance(history, dict) else {}
    projected = history.get("projected_effect_facts", {}) if isinstance(history, dict) else {}
    omit_nullable(committed, ("history", "committed_effects"), ("known_field_union",))
    omit_nullable(projected, ("history", "projected_effect_facts"), ("projected_field_union",))
    for event_index, event in enumerate(source.get("matching_event_history", [])):
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        omit_nullable(payload, ("matching_event_history", event_index, "payload"),
                      ("grant_ref", "approved_artifact_revision", "initial_artifact_revision",
                       "artifact_revision", "revision", "destination", "data_fields"))
    return null_fields, null_list_items


def _restore_nullable_identifiers(state: dict[str, Any], null_fields: list, null_list_items: list) -> None:
    def at_path(root: Any, path: tuple[Any, ...]) -> Any:
        current = root
        for component in path:
            current = current[component]
        return current

    # Outer null groups must be restored before paths into specific groups.
    null_list_items.sort(key=lambda item: len(item[0]))
    for path, indices in null_list_items:
        values = at_path(state, path)
        for index in indices:
            values.insert(index, None)
    for path, key in null_fields:
        at_path(state, path)[key] = None


def transform_model_state(state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Apply the unchanged v1 bijection while preserving nullable evidence exactly."""
    if not isinstance(state, dict) or state.get("schema") != NATIVE_SCHEMA:
        raise ValueError(f"expected {NATIVE_SCHEMA}")
    source = copy.deepcopy(state)
    source["schema"] = V1_NATIVE_SCHEMA
    null_fields, null_list_items = _strip_nullable_identifiers(source)
    transformed, mapping = alpha_bind_v1.transform_model_state(source)
    _restore_nullable_identifiers(transformed, null_fields, null_list_items)
    transformed["schema"] = SCHEMA
    return transformed, mapping


def restore_model_state(state: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """Restore v2 state for audit while keeping sidecar IDs outside model input."""
    if not isinstance(state, dict) or state.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA}")
    source = copy.deepcopy(state)
    source["schema"] = V1_ALPHA_SCHEMA
    null_fields, null_list_items = _strip_nullable_identifiers(source)
    restored = alpha_bind_v1.restore_model_state(source, mapping)
    _restore_nullable_identifiers(restored, null_fields, null_list_items)
    restored["schema"] = NATIVE_SCHEMA
    return restored
