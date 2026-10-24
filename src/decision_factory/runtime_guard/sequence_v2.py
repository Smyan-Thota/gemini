"""Sequence guard input projection v2, preserving the frozen v1 route.

Version 2 adds one factual history field required by the public operation
budget: the number of prior host calls actually committed with ALLOW.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from . import bev
from .host import public_decision_contract

_STATE_DIR = Path(__file__).with_name("sequence_state")
NATIVE_STATE_V2_SHA256 = "4a0a424433a71d758fc1385ab8f4f4069faf11b30e596d70f0a62d50b51e28ab"
ALPHA_BIND_V2_SHA256 = "e45b55bed099eba07a3ba1b18823e6f82c3745a59d8d77d78234cf8cc9895b42"
NATIVE_STATE_V1_SHA256 = "72271c9f99fbaef120279b7ce80753850c6a1650f54423dfc7fb622cb0588b2d"
ALPHA_BIND_V1_SHA256 = "a5acc677620e49e2255915425b30d35c8b3fca639ac4baf22f9c5565534727e6"
NATIVE_SCHEMA = "sequence_guard_native_state/2"
ALPHA_SCHEMA = "sequence_guard_native_state_alpha/2"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_module(name: str, path: Path, expected_sha256: str):
    if not expected_sha256 or _sha256(path) != expected_sha256:
        raise RuntimeError(f"{name} source hash mismatch")
    if str(_STATE_DIR) not in sys.path:
        sys.path.insert(0, str(_STATE_DIR))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"{name} source is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sequence_state_v2(request: dict[str, Any]) -> dict[str, Any]:
    """Return the alpha-bound v2 state for the production Bev encoder."""
    clean = bev._validate_request(request)
    policy = clean.get("host_policy")
    rules = policy.get("workflow_rules") if isinstance(policy, dict) else None
    rule_types = {"release_limit", "approval_status", "joint_disclosure", "recipient_allowlist"}
    if not isinstance(rules, dict) or rules.get("rule_type") not in rule_types:
        raise ValueError("unsupported sequence workflow rule")
    clean["host_policy"]["decision_contract"] = public_decision_contract()

    _load_module("native_state_v1", _STATE_DIR / "native_state_v1.py", NATIVE_STATE_V1_SHA256)
    _load_module("alpha_bind_v1", _STATE_DIR / "alpha_bind_v1.py", ALPHA_BIND_V1_SHA256)
    encoder = _load_module("bev_sequence_native_state_v2",
                           _STATE_DIR / "native_state_v2.py", NATIVE_STATE_V2_SHA256)
    alpha = _load_module("bev_sequence_alpha_bind_v2",
                         _STATE_DIR / "alpha_bind_v2.py", ALPHA_BIND_V2_SHA256)
    native = encoder.encode_native_state_v2(clean)
    model_state, _sidecar = alpha.transform_model_state(native)
    if native.get("schema") != NATIVE_SCHEMA or model_state.get("schema") != ALPHA_SCHEMA:
        raise RuntimeError("sequence v2 transform returned an unsupported schema")
    return model_state
