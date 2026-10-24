"""Routed sequence guard using the audited alpha-bound Bev adapter.

Sequence facts are projected from the host-validated request and identifiers
are bijectively alpha-bound before encoding. The original request remains the
audit evidence; neither labels nor evaluator state enter this module.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from . import bev
from .host import public_decision_contract

SEQUENCE_CHECKPOINT_SHA256 = "57492578fee9ca6cc1d31b0c1a18900e95581d670ab7b4d1a6322b7b88523edf"
LEGACY_V4_CHECKPOINT_SHA256 = "c6c480488f93c1bb09ab2b1c94a75aba0f787d4b4187ba004630d700fa786527"
ALPHA_TRANSFORM_SHA256 = "a5acc677620e49e2255915425b30d35c8b3fca639ac4baf22f9c5565534727e6"
ALPHA_MANIFEST_SHA256 = "d9b7534805f47afeaa3d7b157dc3d802bd997a824b72d1c9c53ff470ae365ca6"
NATIVE_ENCODER_SHA256 = "72271c9f99fbaef120279b7ce80753850c6a1650f54423dfc7fb622cb0588b2d"
SUPPORTED_RULE_TYPES = frozenset({
    "release_limit", "approval_status", "joint_disclosure", "recipient_allowlist",
})
_ROUTE_DIR = Path(__file__).with_name("sequence_state")
_CANDIDATE_PATH = Path(__file__).resolve().parents[3] / "checkpoints/sequence-bev-v7-experimental.pt"
_LEGACY_V4_PATH = Path(__file__).resolve().parents[3] / "checkpoints/legacy-v4-bev.pt"
_encoder = None
_alpha = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_projection():
    global _encoder, _alpha
    if _encoder is None or _alpha is None:
        import importlib.util
        encoder_path = _ROUTE_DIR / "native_state_v1.py"
        alpha_path = _ROUTE_DIR / "alpha_bind_v1.py"
        if _sha256_file(encoder_path) != NATIVE_ENCODER_SHA256:
            raise RuntimeError("sequence native encoder hash mismatch")
        if _sha256_file(alpha_path) != ALPHA_TRANSFORM_SHA256:
            raise RuntimeError("sequence alpha-binding transform hash mismatch")
        spec = importlib.util.spec_from_file_location("bev_sequence_native_state", encoder_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("sequence native encoder is unavailable")
        _encoder = importlib.util.module_from_spec(spec)
        import sys
        if str(_ROUTE_DIR) not in sys.path:
            sys.path.insert(0, str(_ROUTE_DIR))
        sys.modules[spec.name] = _encoder
        spec.loader.exec_module(_encoder)
        spec = importlib.util.spec_from_file_location("bev_sequence_alpha_bind", alpha_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("sequence alpha-binding transform is unavailable")
        _alpha = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = _alpha
        spec.loader.exec_module(_alpha)
    return _encoder, _alpha


def sequence_state(request: dict[str, Any]) -> dict[str, Any]:
    """Build the exact alpha-bound state expected by the frozen V7 adapter."""
    clean = bev._validate_request(request)
    clean["host_policy"]["decision_contract"] = public_decision_contract()
    rules = clean["host_policy"].get("workflow_rules")
    if not isinstance(rules, dict) or rules.get("rule_type") not in SUPPORTED_RULE_TYPES:
        raise ValueError("unsupported sequence workflow rule")
    encoder, alpha = _load_projection()
    native = encoder.encode_native_state(clean)
    projected_facts = native.get("history", {}).get("projected_effect_facts", {})
    # For a non-export proposal (for example read_file), projected_field_union
    # is explicitly null: there is no proposed disclosure to union. The frozen
    # alpha binder collects only opaque identifiers and expects present lists,
    # so omit this null only during identifier collection, then restore it
    # byte-for-value in the model state. No facts or authorization labels are
    # inferred here.
    null_projected_union = (
        isinstance(projected_facts, dict)
        and "projected_field_union" in projected_facts
        and projected_facts["projected_field_union"] is None
    )
    if null_projected_union:
        projected_facts.pop("projected_field_union")
    state, _sidecar = alpha.transform_model_state(native)
    if null_projected_union:
        state["history"]["projected_effect_facts"]["projected_field_union"] = None
    if state.get("schema") != "sequence_guard_native_state_alpha/1":
        raise RuntimeError("sequence state transform returned an unsupported schema")
    return state


def evaluate_sequence_proposal(request: dict[str, Any], *,
                               head_checkpoint: str | Path = _CANDIDATE_PATH,
                               device: str = "cuda") -> dict[str, Any]:
    """Assess a sequence call using CUDA FP32; all errors fail closed to REVIEW."""
    started = time.perf_counter()
    clean = None
    effective_device = None
    checkpoint_sha = None
    stage = "request_validation"
    error_code = "invalid_request"
    try:
        clean = bev._validate_request(request)
        rules = clean["host_policy"].get("workflow_rules")
        if not isinstance(rules, dict) or rules.get("rule_type") not in SUPPORTED_RULE_TYPES:
            stage, error_code = "request_validation", "unsupported_workflow_rule"
            raise bev.GuardInputError("unsupported sequence workflow rule")
        if device != "cuda":
            stage, error_code = "device_validation", "unsupported_device"
            raise bev.GuardInputError("the sequence Bev profile supports CUDA FP32 only")
        stage, error_code = "device_validation", "cuda_unavailable"
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for the sequence guard but is unavailable")
        effective_device = "cuda"
        stage, error_code = "request_contract", "decision_contract_unavailable"
        clean["host_policy"]["decision_contract"] = public_decision_contract()
        payload_bytes = bev._json_bytes(clean)
        request_sha256 = hashlib.sha256(payload_bytes).hexdigest()
        stage, error_code = "checkpoint_validation", "checkpoint_unavailable"
        checkpoint = Path(head_checkpoint).expanduser().resolve(strict=True)
        stage, error_code = "checkpoint_validation", "checkpoint_hash_mismatch"
        checkpoint_sha = _sha256_file(checkpoint)
        if checkpoint_sha != SEQUENCE_CHECKPOINT_SHA256:
            raise RuntimeError("sequence checkpoint hash mismatch")
        stage, error_code = "checkpoint_contract", "checkpoint_contract_mismatch"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        expected_metadata = {
            "training_input_mode": "stored_alpha_bound_state_no_runtime_retransform",
            "model_state_schema": "sequence_guard_native_state_alpha/1",
            "model_state_transform_sha256": ALPHA_TRANSFORM_SHA256,
            "model_state_manifest_sha256": ALPHA_MANIFEST_SHA256,
        }
        if any(payload.get(key) != value for key, value in expected_metadata.items()):
            raise RuntimeError("sequence checkpoint input contract mismatch")
        stage, error_code = "state_projection", "state_projection_failure"
        model_state = sequence_state(clean)
        stage, error_code = "model_load", "model_load_failure"
        decider, checkpoint_sha = bev._load_candidate_decider(checkpoint)
        if decider.device.type != device:
            decider.network.to(torch.device(device))
            decider.device = torch.device(device)
            decider.precision = "fp32"
            decider.network.eval()
        stage, error_code = "input_validation", "input_limit_exceeded"
        state_tokens, option_tokens, input_tokens = bev._measure_tokens(decider, model_state)
        stage, error_code = "inference", "model_inference_failure"
        raw = decider.decide(model_state, {"guard": bev._QUESTION})["guard"]
        stage, error_code = "output_validation", "invalid_model_output"
        if not isinstance(raw, dict) or raw.get("type") != "choice":
            raise RuntimeError("Bev returned a malformed sequence choice")
        selected = raw.get("choice")
        probabilities = raw.get("probabilities")
        if selected not in bev.CHOICES or not isinstance(probabilities, dict) or set(probabilities) != set(bev.CHOICES):
            raise RuntimeError("Bev returned an invalid sequence choice or scores")
        scores = {key: float(value) for key, value in probabilities.items()}
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores.values()) or abs(sum(scores.values()) - 1) > 0.02:
            raise RuntimeError("Bev returned invalid sequence probabilities")
        return {
            "schema": bev.ASSESSMENT_SCHEMA, "status": "assessed", "disposition": selected,
            "scores": scores, "selected": selected,
            "evidence": {
                "request": clean, "request_sha256": request_sha256,
                "model_id": bev.MODEL_ID, "revision": bev.MODEL_REVISION,
                "package_version": bev.version("bev-decider"), "device": device,
                "profile": "sequence_cuda_fp32", "requested_device": device,
                "effective_device": effective_device, "failure_stage": None,
                "error_code": None,
                "model_state": model_state, "question": bev._QUESTION,
                "state_tokens": state_tokens, "max_state_tokens": bev.MAX_STATE_TOKENS,
                "option_tokens": option_tokens, "max_option_tokens": bev.MAX_OPTION_TOKENS,
                "input_tokens": input_tokens, "max_input_tokens": bev.MAX_INPUT_TOKENS,
                "latency_ms": max(0, int((time.perf_counter() - started) * 1000)),
                "cache_hit": False, "model_latency_ms": max(0, int((time.perf_counter() - started) * 1000)),
            "raw_choice": raw, "head_checkpoint_sha256": checkpoint_sha,
                "input_transform_sha256": ALPHA_TRANSFORM_SHA256, "error": None,
            },
        }
    except Exception as exc:
        if clean is not None:
            clean = json.loads(bev._json_bytes(clean).decode("utf-8"))
            if stage not in {"device_validation"}:
                try:
                    clean["host_policy"]["decision_contract"] = public_decision_contract()
                except Exception:
                    pass
        else:
            try:
                clean = bev._validate_request(request)
            except Exception:
                clean = None
        serialized = None
        if clean is not None:
            try:
                serialized = bev._json_bytes(clean)
            except Exception:
                pass
        evidence_request_hash = hashlib.sha256(serialized).hexdigest() if serialized is not None else None
        return {
            "schema": bev.ASSESSMENT_SCHEMA, "status": "review", "disposition": "REVIEW",
            "scores": {key: None for key in bev.CHOICES}, "selected": None,
            "evidence": {
                "request": clean, "request_sha256": evidence_request_hash,
                "model_id": bev.MODEL_ID, "revision": bev.MODEL_REVISION,
                "profile": "sequence_cuda_fp32", "device": effective_device or device,
                "requested_device": device, "effective_device": effective_device,
                "head_checkpoint_sha256": checkpoint_sha, "input_transform_sha256": ALPHA_TRANSFORM_SHA256,
                "failure_stage": stage, "error_code": error_code,
                "error": f"{error_code}: {type(exc).__name__}: {exc}",
                "latency_ms": max(0, int((time.perf_counter() - started) * 1000)),
                "cache_hit": False, "model_latency_ms": None,
            },
        }


def evaluate_routed_proposal(request: dict[str, Any], *,
                             sequence_checkpoint: str | Path = _CANDIDATE_PATH,
                             legacy_checkpoint: str | Path = _LEGACY_V4_PATH,
                             sequence_device: str = "cuda") -> dict[str, Any]:
    """Route four known public workflows to V7; retain the legacy V4 path otherwise.

    Missing workflow metadata is the established legacy contract. Present but
    malformed/unknown workflow metadata fails closed to REVIEW.
    """
    try:
        clean = bev._validate_request(request)
    except Exception:
        return bev.evaluate_proposal(request, head_checkpoint=legacy_checkpoint)
    policy = clean.get("host_policy", {})
    if "workflow_rules" not in policy:
        try:
            legacy_sha = _sha256_file(Path(legacy_checkpoint).expanduser().resolve(strict=True))
        except Exception:
            return bev._review(request=clean, reason="legacy_v4_checkpoint_unavailable",
                               started=time.perf_counter(), requested_device="cpu",
                               effective_device=None, profile="legacy_v4_cpu",
                               failure_stage="checkpoint_validation", error_code="checkpoint_unavailable")
        if legacy_sha != LEGACY_V4_CHECKPOINT_SHA256:
            return bev._review(request=clean, reason="legacy_v4_checkpoint_hash_mismatch",
                               started=time.perf_counter(), requested_device="cpu",
                               effective_device=None, profile="legacy_v4_cpu",
                               failure_stage="checkpoint_validation", error_code="checkpoint_hash_mismatch")
        return bev.evaluate_proposal(clean, head_checkpoint=legacy_checkpoint, device="cpu")
    rules = policy.get("workflow_rules")
    if not isinstance(rules, dict) or rules.get("rule_type") not in SUPPORTED_RULE_TYPES:
        return evaluate_sequence_proposal(clean, head_checkpoint=sequence_checkpoint, device=sequence_device)
    return evaluate_sequence_proposal(clean, head_checkpoint=sequence_checkpoint, device=sequence_device)
