"""Operation-facts v3 sequence state profile, separate from frozen V2."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from . import bev
from .host import public_decision_contract
from .sequence_state import alpha_bind_operation_facts_v3, operation_facts_v3

_STATE_DIR = Path(__file__).with_name("sequence_state")
NATIVE_STATE_V3_SHA256 = "6d61b96a7efc589e6c763bd955a8b1e61be687f3f3acbdf2a2f8efd614fdf072"
ALPHA_BIND_V3_SHA256 = "6a4723523845ca1579fb36d9d61f86bbeba215ebb8855b1106682542592a49b2"
ALPHA_BIND_V1_SHA256 = "a5acc677620e49e2255915425b30d35c8b3fca639ac4baf22f9c5565534727e6"
NATIVE_SCHEMA = operation_facts_v3.NATIVE_SCHEMA
ALPHA_SCHEMA = alpha_bind_operation_facts_v3.SCHEMA
CONTRACT_SHA256 = "5f6db91e4a51c5650897a7307ab4eb235d2dcb770896a313de7e8a7d4bb4612f"
PUBLIC_SEMANTICS_SHA256 = "a3aab29b60b09a38d87081314a3c53a7c23815da5e7fd161e254d1124f1cdaf5"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_pins() -> dict[str, str]:
    """Current source pins included in candidate metadata and assessor evidence."""
    contract_path = Path(__file__).resolve().parents[3] / "docs" / "COMPACT_OPERATION_FACTS_V3_CONTRACT.md"
    actual_contract_sha = _sha256_file(contract_path)
    if actual_contract_sha != CONTRACT_SHA256:
        raise RuntimeError("operation-facts v3 contract hash mismatch")
    public_semantics_path = Path(__file__).resolve().parents[3] / "docs" / "PUBLIC_DECISION_SEMANTICS.txt"
    actual_public_semantics_sha = _sha256_file(public_semantics_path)
    if actual_public_semantics_sha != PUBLIC_SEMANTICS_SHA256:
        raise RuntimeError("public decision semantics hash mismatch")
    native_sha = _sha256_file(_STATE_DIR / "operation_facts_v3.py")
    alpha_sha = _sha256_file(_STATE_DIR / "alpha_bind_operation_facts_v3.py")
    alpha_base_sha = _sha256_file(_STATE_DIR / "alpha_bind_v1.py")
    if native_sha != NATIVE_STATE_V3_SHA256 or alpha_sha != ALPHA_BIND_V3_SHA256:
        raise RuntimeError("operation-facts v3 implementation source hash mismatch")
    if alpha_base_sha != ALPHA_BIND_V1_SHA256:
        raise RuntimeError("pinned alpha base source hash mismatch")
    pins = {
        "operation_facts_v3_sha256": native_sha,
        "alpha_bind_operation_facts_v3_sha256": alpha_sha,
        "alpha_bind_v1_sha256": alpha_base_sha,
        "contract_sha256": actual_contract_sha,
        "public_semantics_sha256": actual_public_semantics_sha,
    }
    return pins


def native_state_v3(request: dict[str, Any]) -> dict[str, Any]:
    """Return native typed operation facts for one validated public request."""
    clean = bev._validate_request(request)
    policy = clean.get("host_policy")
    if not isinstance(policy, dict):
        raise ValueError("host_policy must be an object")
    policy["decision_contract"] = public_decision_contract()
    native = operation_facts_v3.encode_operation_facts_v3(clean)
    if native.get("schema") != NATIVE_SCHEMA:
        raise RuntimeError("operation-facts v3 returned an unsupported native schema")
    return native


def sequence_state_v3(request: dict[str, Any]) -> dict[str, Any]:
    """Return alpha-bound operation facts for the candidate model input."""
    native = native_state_v3(request)
    model_state, _sidecar = alpha_bind_operation_facts_v3.transform_operation_facts_v3(native)
    if model_state.get("schema") != ALPHA_SCHEMA:
        raise RuntimeError("operation-facts v3 transform returned an unsupported schema")
    return model_state
