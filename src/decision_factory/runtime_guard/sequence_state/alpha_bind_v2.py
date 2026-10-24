"""Alpha binder v2 preserves v1 identifier mapping and native-state v2 facts."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import alpha_bind_v1  # noqa: E402

NATIVE_SCHEMA = "sequence_guard_native_state/2"
SCHEMA = "sequence_guard_native_state_alpha/2"
V1_NATIVE_SCHEMA = "sequence_guard_native_state/1"
V1_ALPHA_SCHEMA = "sequence_guard_native_state_alpha/1"


def transform_model_state(state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Apply the unchanged v1 bijection to v2 facts without changing them."""
    if not isinstance(state, dict) or state.get("schema") != NATIVE_SCHEMA:
        raise ValueError(f"expected {NATIVE_SCHEMA}")
    source = copy.deepcopy(state)
    source["schema"] = V1_NATIVE_SCHEMA
    projected = source.get("history", {}).get("projected_effect_facts", {})
    null_projected_union = (
        isinstance(projected, dict)
        and "projected_field_union" in projected
        and projected["projected_field_union"] is None
    )
    if null_projected_union:
        projected.pop("projected_field_union")
    transformed, mapping = alpha_bind_v1.transform_model_state(source)
    if null_projected_union:
        transformed["history"]["projected_effect_facts"]["projected_field_union"] = None
    transformed["schema"] = SCHEMA
    return transformed, mapping


def restore_model_state(state: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """Restore v2 state for audit while keeping sidecar IDs outside model input."""
    if not isinstance(state, dict) or state.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA}")
    source = copy.deepcopy(state)
    source["schema"] = V1_ALPHA_SCHEMA
    restored = alpha_bind_v1.restore_model_state(source, mapping)
    restored["schema"] = NATIVE_SCHEMA
    return restored
