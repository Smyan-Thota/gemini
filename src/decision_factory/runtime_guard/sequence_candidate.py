"""Opt-in assessor for operator-pinned alpha-v2 sequence checkpoints.

The default routed runtime remains the frozen V7/V4 path. This module only
loads a candidate when the caller supplies its exact expected file digest.
Inference is deliberately singleton CUDA FP32: padding peers have been shown
to change sequence decisions.
"""
from __future__ import annotations

import hashlib
import json
import math
import gc
from pathlib import Path
import re
import time
from typing import Any, Mapping

from . import bev, sequence_v2
from .host import public_decision_contract


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TRAINING_INPUT_MODE = "production_native_v2_token_stream_singleton_fp32"
_ALPHA_STATE_SCHEMA = "sequence_guard_native_state_alpha/2"
_CANDIDATE_SCHEMA = "bev_lora_head_checkpoint/2"
_V9_PRIOR_SHA256 = "ec3b8ef8fddc4b5560e25149e5d0c98999b6046e58c187e116cf26bc8aeab7c2"
_V9_PRIOR_MANIFEST_SHA256 = "254026dbcc2f8118d78e44c6dbaf5957bb10315f392416e02a131799a6a82359"
_V9_PRIOR_CONTRACT = "sequence-guard-v9-native-v2-full-depth-balanced-source-sequence-only"
_NETWORK_DIGESTS: dict[str, tuple[str, str]] = {}


class CandidateAssessmentError(RuntimeError):
    """An assessor failure with stable machine-readable stage and code."""

    def __init__(self, failure_stage: str, error_code: str, message: str):
        super().__init__(message)
        self.failure_stage = failure_stage
        self.error_code = error_code


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tensor(digest: Any, name: str, tensor: Any) -> None:
    value = tensor.detach().to(device="cpu").contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(_canonical_bytes(list(value.shape)))
    # Viewing bytes also supports bfloat16 tensors without dtype conversion.
    raw = value.view(__import__("torch").uint8).reshape(-1).numpy()
    view = memoryview(raw)
    chunk_size = 16 * 1024 * 1024
    for start in range(0, len(view), chunk_size):
        digest.update(view[start:start + chunk_size])


def _state_mapping_digest(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        _sha256_tensor(digest, name, tensor)
    return digest.hexdigest()


def _loaded_network_digests(decider: Any, checkpoint_sha256: str) -> tuple[str, str]:
    cached = _NETWORK_DIGESTS.get(checkpoint_sha256)
    if cached is not None:
        return cached
    full = _state_mapping_digest(decider.network.state_dict())
    backbone_state = decider.network.backbone.state_dict()
    frozen_backbone = {name: tensor for name, tensor in backbone_state.items()
                       if "lora_" not in name}
    frozen = _state_mapping_digest(frozen_backbone)
    result = (full, frozen)
    _NETWORK_DIGESTS[checkpoint_sha256] = result
    return result


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value.lower()) is not None


def _validate_checkpoint_metadata(payload: Any, *, checkpoint_sha256: str,
                                  allow_unpromoted_v9_prior: bool) -> tuple[dict[str, Any], str, bool]:
    if not isinstance(payload, dict) or payload.get("schema") != _CANDIDATE_SCHEMA:
        raise CandidateAssessmentError("checkpoint_contract", "unsupported_checkpoint_schema",
                                       "candidate must use bev_lora_head_checkpoint/2")
    expected_base = {
        "base_model_id": bev.MODEL_ID,
        "base_revision": bev.MODEL_REVISION,
        "base_weight_sha256": bev.MODEL_SHA256,
    }
    if any(payload.get(key) != value for key, value in expected_base.items()):
        raise CandidateAssessmentError("checkpoint_contract", "base_model_mismatch",
                                       "candidate is not bound to the pinned Bev base")
    is_v9_prior = (allow_unpromoted_v9_prior and checkpoint_sha256 == _V9_PRIOR_SHA256)
    if is_v9_prior:
        if (payload.get("training_input_mode") != "stored_alpha_v2_state_no_runtime_retransform"
                or payload.get("candidate_training_contract") != _V9_PRIOR_CONTRACT
                or payload.get("model_state_manifest_sha256") != _V9_PRIOR_MANIFEST_SHA256
                or payload.get("model_state_independent_audit_sha256") is not None):
            raise CandidateAssessmentError("checkpoint_contract", "v9_prior_metadata_mismatch",
                                           "the pinned V9 prior metadata does not match its historical receipt")
    elif payload.get("training_input_mode") != _TRAINING_INPUT_MODE:
        raise CandidateAssessmentError("checkpoint_contract", "input_mode_mismatch",
                                       "candidate must use production-native v2 singleton FP32 input serialization")
    if payload.get("model_state_schema") != _ALPHA_STATE_SCHEMA:
        raise CandidateAssessmentError("checkpoint_contract", "model_state_schema_mismatch",
                                       "candidate must be trained on sequence_guard_native_state_alpha/2")
    if (not is_v9_prior
            and payload.get("model_state_encoder_sha256") != sequence_v2.NATIVE_STATE_V2_SHA256):
        raise CandidateAssessmentError("checkpoint_contract", "native_encoder_mismatch",
                                       "candidate native-state encoder pin does not match runtime")
    if payload.get("model_state_transform_sha256") != sequence_v2.ALPHA_BIND_V2_SHA256:
        raise CandidateAssessmentError("checkpoint_contract", "alpha_transform_mismatch",
                                       "candidate alpha-binding transform pin does not match runtime")
    if not is_v9_prior:
        for field in ("model_state_manifest_sha256", "model_state_independent_audit_sha256"):
            if not _valid_sha(payload.get(field)):
                raise CandidateAssessmentError("checkpoint_contract", f"invalid_{field}",
                                               f"candidate metadata requires a valid {field}")
    selected_step = payload.get("selected_step")
    if type(selected_step) is not int or selected_step <= 0:
        raise CandidateAssessmentError("checkpoint_contract", "invalid_selected_step",
                                       "candidate selected_step must identify a trained checkpoint")
    return payload, ("unpromoted_warm_start_prior" if is_v9_prior else "candidate"), not is_v9_prior


def _failure(*, request: dict[str, Any] | None, request_sha256: str | None,
             expected_sha256: str | None, actual_sha256: str | None,
             started: float, requested_device: str, effective_device: str | None,
             failure_stage: str, error_code: str, error: str) -> dict[str, Any]:
    return {
        "schema": bev.ASSESSMENT_SCHEMA,
        "status": "review",
        "disposition": "REVIEW",
        "scores": {key: None for key in bev.CHOICES},
        "selected": None,
        "evidence": {
            "request": request,
            "request_sha256": request_sha256,
            "model_id": bev.MODEL_ID,
            "revision": bev.MODEL_REVISION,
            "profile": "sequence_candidate_cuda_fp32_singleton",
            "requested_device": requested_device,
            "effective_device": effective_device,
            "precision": "fp32",
            "tf32": False,
            "inference_batch_size": 1,
            "checkpoint_expected_sha256": expected_sha256,
            "checkpoint_actual_sha256": actual_sha256,
            "failure_stage": failure_stage,
            "error_code": error_code,
            "error": error,
            "latency_ms": max(0, int((time.perf_counter() - started) * 1000)),
            "cache_hit": False,
            "model_latency_ms": None,
        },
    }


def assess_candidate(request: dict[str, Any], *, checkpoint_path: str | Path,
                     expected_checkpoint_sha256: str, device: str = "cuda",
                     allow_unpromoted_v9_prior: bool = False) -> dict[str, Any]:
    """Assess one request with an explicitly SHA-pinned alpha-v2 candidate.

    No candidate path or digest is implicit. CPU is rejected before checkpoint
    loading; successful inference uses exactly one encoded request and FP32.
    Failures return a typed REVIEW assessment and never force a host verdict.
    """
    started = time.perf_counter()
    clean = None
    request_sha256 = None
    actual_sha256 = None
    effective_device = None
    stage, code = "request_validation", "invalid_request"
    try:
        clean = bev._validate_request(request)
        clean["host_policy"]["decision_contract"] = public_decision_contract()
        request_sha256 = hashlib.sha256(_canonical_bytes(clean)).hexdigest()

        stage, code = "checkpoint_validation", "invalid_expected_checkpoint_sha256"
        if not _valid_sha(expected_checkpoint_sha256):
            raise CandidateAssessmentError(stage, code, "expected checkpoint SHA256 must be 64 hex characters")
        expected_checkpoint_sha256 = expected_checkpoint_sha256.lower()

        stage, code = "device_validation", "unsupported_device"
        if device != "cuda":
            raise CandidateAssessmentError(stage, code, "sequence candidates support CUDA FP32 only")
        stage, code = "device_validation", "cuda_unavailable"
        import torch
        if not torch.cuda.is_available():
            raise CandidateAssessmentError(stage, code, "CUDA is unavailable")
        effective_device = "cuda"

        stage, code = "checkpoint_validation", "checkpoint_unavailable"
        checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
        if not checkpoint.is_file():
            raise CandidateAssessmentError(stage, code, "candidate checkpoint is not a regular file")
        actual_sha256 = _sha256_file(checkpoint)
        stage, code = "checkpoint_validation", "checkpoint_hash_mismatch"
        if actual_sha256 != expected_checkpoint_sha256:
            raise CandidateAssessmentError(stage, code, "candidate checkpoint SHA256 does not match operator pin")

        stage, code = "checkpoint_contract", "checkpoint_metadata_invalid"
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise CandidateAssessmentError(stage, code, f"safe checkpoint read failed: {type(exc).__name__}") from exc
        metadata, checkpoint_role, promotion_eligible = _validate_checkpoint_metadata(
            payload, checkpoint_sha256=actual_sha256,
            allow_unpromoted_v9_prior=allow_unpromoted_v9_prior)
        for key in ("lora_state_dict", "head_state_dict"):
            state_dict = payload.get(key)
            if not isinstance(state_dict, dict) or not state_dict:
                raise CandidateAssessmentError("checkpoint_contract", "missing_candidate_state",
                                               f"candidate is missing {key}")
            if any(hasattr(value, "is_floating_point") and value.is_floating_point()
                   and value.dtype != torch.float32 for value in state_dict.values()):
                raise CandidateAssessmentError("checkpoint_contract", "non_fp32_checkpoint_state",
                                               f"candidate {key} must be stored in FP32")

        stage, code = "state_projection", "state_projection_failure"
        model_state = sequence_v2.sequence_state_v2(clean)
        stage, code = "input_validation", "input_limit_exceeded"
        decider, loaded_sha256 = bev._load_candidate_decider(checkpoint)
        if loaded_sha256 != expected_checkpoint_sha256:
            raise CandidateAssessmentError("model_load", "loaded_checkpoint_mismatch",
                                           "Bev loader returned a different checkpoint identity")
        decider.network.to(device=torch.device("cuda"), dtype=torch.float32)
        decider.device = torch.device("cuda")
        decider.precision = "fp32"
        decider.network.eval()
        for parameter in decider.network.parameters():
            if parameter.is_floating_point() and parameter.dtype != torch.float32:
                raise CandidateAssessmentError("model_load", "non_fp32_parameter",
                                               "candidate network contains a non-FP32 parameter")
        for buffer in decider.network.buffers():
            if buffer.is_floating_point() and buffer.dtype != torch.float32:
                raise CandidateAssessmentError("model_load", "non_fp32_buffer",
                                               "candidate network contains a non-FP32 buffer")
        if decider.network.training:
            raise CandidateAssessmentError("model_load", "network_not_eval",
                                           "candidate network did not enter evaluation mode")
        # Cache complete parameter/buffer identities once for each immutable
        # checkpoint, after the exact FP32 production casting path.
        network_state_sha256, frozen_backbone_sha256 = _loaded_network_digests(decider, loaded_sha256)

        question = bev._QUESTION
        question_text = json.dumps(question, ensure_ascii=False, indent=2)
        model_state_text = json.dumps(model_state, ensure_ascii=False, indent=2)
        state_tokens = len(decider.encoder.tok.encode(model_state_text))
        option_tokens = {key: len(decider.encoder.tok.encode(f"{key}: {description}"))
                         for key, description in bev._CHOICE_DESCRIPTIONS.items()}
        if state_tokens > bev.MAX_STATE_TOKENS:
            raise CandidateAssessmentError("input_validation", "state_limit_exceeded",
                                           "model state exceeds the encoder context bound")
        if any(count > bev.MAX_OPTION_TOKENS for count in option_tokens.values()):
            raise CandidateAssessmentError("input_validation", "option_limit_exceeded",
                                           "fixed decision option exceeds the encoder context bound")
        from bev_decider.encode import question_options
        rendered_options, rendered_keys = question_options(question)
        encoded = decider.encoder.encode(model_state, question)
        state_text_sha256 = hashlib.sha256(model_state_text.encode("utf-8")).hexdigest()
        input_ids = [int(token) for token in encoded["input_ids"]]
        if not input_ids:
            raise CandidateAssessmentError("input_encoding", "empty_encoded_input", "encoded request is empty")
        input_ids_sha256 = hashlib.sha256(_canonical_bytes(input_ids)).hexdigest()
        mask = encoded["attention_mask"].detach().cpu().contiguous()
        mask_digest = hashlib.sha256()
        _sha256_tensor(mask_digest, "attention_mask", mask)
        attention_mask_sha256 = mask_digest.hexdigest()
        encoded_example_sha256 = hashlib.sha256(_canonical_bytes({
            "input_ids": input_ids,
            "position_ids": [int(value) for value in encoded["position_ids"]],
            "choice_read_idx": [int(value) for value in encoded["choice_read_idx"]],
            "answer_end_idx": int(encoded["answer_end_idx"]),
            "task_type": int(encoded["task_type"]),
            "keys": list(encoded["keys"]),
            "attention_mask_sha256": attention_mask_sha256,
        })).hexdigest()
        if len(input_ids) > bev.MAX_INPUT_TOKENS:
            raise CandidateAssessmentError("input_validation", "input_limit_exceeded",
                                           "encoded request exceeds the supported input length")

        stage, code = "inference", "model_inference_failure"
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        old_precision = torch.get_float32_matmul_precision()
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            # The production Bev wrapper collates one example and performs one
            # network forward. Keeping this call singleton avoids pad peers.
            with torch.inference_mode():
                probabilities = decider.probabilities([encoded])
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
            torch.set_float32_matmul_precision(old_precision)
        if probabilities.ndim != 2 or probabilities.shape[0] != 1:
            raise CandidateAssessmentError("output_validation", "non_singleton_output",
                                           "candidate returned a non-singleton result")
        values = probabilities[0, :len(encoded["keys"])].tolist()
        scores = {key: float(value) for key, value in zip(encoded["keys"], values)}
        if set(scores) != set(bev.CHOICES) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
            raise CandidateAssessmentError("output_validation", "invalid_probabilities",
                                           "candidate returned invalid choice probabilities")
        if abs(sum(scores.values()) - 1.0) > 0.02:
            raise CandidateAssessmentError("output_validation", "invalid_probabilities",
                                           "candidate probabilities do not sum to one")
        selected = max(encoded["keys"], key=lambda key: scores[key])
        elapsed = max(0, int((time.perf_counter() - started) * 1000))
        source_hash = _sha256_file(Path(__file__).resolve())
        return {
            "schema": bev.ASSESSMENT_SCHEMA,
            "status": "assessed",
            "disposition": selected,
            "scores": scores,
            "selected": selected,
            "evidence": {
                "request": clean,
                "request_sha256": request_sha256,
                "model_id": bev.MODEL_ID,
                "revision": bev.MODEL_REVISION,
                "base_weight_sha256": bev.MODEL_SHA256,
                "package_version": bev.version("bev-decider"),
                "profile": "sequence_candidate_cuda_fp32_singleton",
                "requested_device": device,
                "effective_device": "cuda",
                "precision": "fp32",
                "tf32": False,
                "inference_batch_size": 1,
                "checkpoint_path": str(checkpoint),
                "checkpoint_expected_sha256": expected_checkpoint_sha256,
                "checkpoint_sha256": loaded_sha256,
                "checkpoint_schema": metadata["schema"],
                "checkpoint_role": checkpoint_role,
                "promotion_eligible": promotion_eligible,
                "training_input_mode": metadata.get("training_input_mode"),
                "selected_step": metadata["selected_step"],
                "model_state_schema": metadata["model_state_schema"],
                "model_state_encoder_sha256": metadata.get("model_state_encoder_sha256"),
                "runtime_native_encoder_sha256": sequence_v2.NATIVE_STATE_V2_SHA256,
                "model_state_transform_sha256": metadata["model_state_transform_sha256"],
                "model_state_manifest_sha256": metadata["model_state_manifest_sha256"],
                "model_state_independent_audit_sha256": metadata["model_state_independent_audit_sha256"],
                "loaded_network_state_sha256": network_state_sha256,
                "frozen_backbone_state_sha256": frozen_backbone_sha256,
                "model_state_sha256": hashlib.sha256(_canonical_bytes(model_state)).hexdigest(),
                "model_state_text": model_state_text,
                "model_state_text_sha256": state_text_sha256,
                "question_text": question_text,
                "question_sha256": hashlib.sha256(question_text.encode("utf-8")).hexdigest(),
                "rendered_option_keys": list(rendered_keys),
                "rendered_options": list(rendered_options),
                "options_sha256": hashlib.sha256(_canonical_bytes(rendered_options)).hexdigest(),
                "encoded_input_sha256": encoded_example_sha256,
                "input_ids_sha256": input_ids_sha256,
                "attention_mask_sha256": attention_mask_sha256,
                "input_tokens": len(input_ids),
                "state_tokens": state_tokens,
                "option_tokens": option_tokens,
                "max_state_tokens": bev.MAX_STATE_TOKENS,
                "max_input_tokens": bev.MAX_INPUT_TOKENS,
                "candidate_assessor_sha256": source_hash,
                "sequence_v2_source_sha256": _sha256_file(Path(sequence_v2.__file__).resolve()),
                "native_encoder_sha256": sequence_v2.NATIVE_STATE_V2_SHA256,
                "alpha_transform_sha256": sequence_v2.ALPHA_BIND_V2_SHA256,
                "failure_stage": None,
                "error_code": None,
                "error": None,
                "latency_ms": elapsed,
                "cache_hit": False,
                "model_latency_ms": elapsed,
            },
        }
    except CandidateAssessmentError as exc:
        stage, code = exc.failure_stage, exc.error_code
        error = f"{code}: {exc}"
    except Exception as exc:
        error = f"{code}: {type(exc).__name__}: {exc}"
    if clean is not None and request_sha256 is None:
        try:
            request_sha256 = hashlib.sha256(_canonical_bytes(clean)).hexdigest()
        except Exception:
            pass
    return _failure(request=clean, request_sha256=request_sha256,
                    expected_sha256=expected_checkpoint_sha256 if isinstance(expected_checkpoint_sha256, str) else None,
                    actual_sha256=actual_sha256, started=started,
                    requested_device=device if isinstance(device, str) else "invalid",
                    effective_device=effective_device, failure_stage=stage,
                    error_code=code, error=error)


def release_candidate_cache(checkpoint_path: str | Path, *,
                            expected_checkpoint_sha256: str,
                            preserve_network: Any = None) -> dict[str, Any]:
    """Release an assessor-loaded CUDA network after a completed evaluation.

    Training-time selection keeps its training model resident. Passing that
    exact network makes this helper a safe no-op when the requested checkpoint
    is the warm-start checkpoint currently being optimized. Callers must ensure
    no assessment using this cache entry is in flight while releasing it.
    """
    if not _valid_sha(expected_checkpoint_sha256):
        raise ValueError("expected checkpoint SHA256 must be 64 hexadecimal characters")
    checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
    actual = _sha256_file(checkpoint)
    expected = expected_checkpoint_sha256.lower()
    if actual != expected:
        raise ValueError("checkpoint does not match expected SHA256")

    cached = bev._candidate_deciders.get(actual)
    if cached is None:
        return {"status": "not_cached", "checkpoint_sha256": actual,
                "released": False, "preserved_training_model": False}
    if preserve_network is not None and getattr(cached, "network", None) is preserve_network:
        return {"status": "preserved_training_model", "checkpoint_sha256": actual,
                "released": False, "preserved_training_model": True}

    if bev._candidate_deciders.get(actual) is cached:
        del bev._candidate_deciders[actual]
    network = getattr(cached, "network", None)
    if network is not None:
        network.to(device="cpu")
    del cached
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # Cache cleanup is best-effort after the CPU move; it must not alter
        # or reinterpret the assessment that has already completed.
        pass
    return {"status": "released", "checkpoint_sha256": actual,
            "released": True, "preserved_training_model": False}
