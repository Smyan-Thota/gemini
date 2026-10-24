"""Pinned frozen Bev features and a small authorization readout.

The backbone remains Bev, including its upstream adapters. No reference policy
or training labels are imported here. Every inference call is singleton FP32.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from . import bev

FEATURE_PROFILE = "bev-last-middle-option-answer-v1"
SCHEMA = "bev_frozen_readout/1"
CHOICES = ("PERMIT", "BLOCK", "REVIEW")


def encode_scene(decider, scene):
    """Encode a factual scene without silently truncating the state."""
    if not isinstance(scene, dict):
        raise ValueError("scene must be an object")
    text = json.dumps(scene, ensure_ascii=False, indent=2, allow_nan=False)
    if len(decider.encoder.tok.encode(text)) > decider.encoder.max_state_tokens:
        raise ValueError("scene exceeds supported context; review required")
    encoded = decider.encoder.encode(scene, bev._QUESTION)
    if tuple(encoded["keys"]) != CHOICES:
        raise ValueError("unexpected class order")
    return encoded


def extract_features(decider, encoded, device="cuda"):
    """Return middle/final answer and option features plus original logits."""
    import torch
    batch = {key: value.to(device) for key, value in decider.encoder.collate([encoded]).items()}
    with torch.inference_mode():
        out = decider.network.backbone(
            input_ids=batch["input_ids"], position_ids=batch["position_ids"],
            attention_mask=batch["attention_mask"], use_cache=False,
            output_hidden_states=True)
        middle = out.hidden_states[len(out.hidden_states) // 2].float()
        last = out.last_hidden_state.float()
        idx = batch["choice_read_idx"][0]
        answer = int(batch["answer_end_idx"][0])
        features = torch.cat([middle[0, answer], last[0, answer],
                              middle[0, idx].flatten(), last[0, idx].flatten()])
        logits = decider.network.head(last[:, idx], last[:, answer],
                                     batch["choice_mask"], batch["task_type"])[0]
    if not torch.isfinite(features).all() or not torch.isfinite(logits).all():
        raise ValueError("nonfinite model features")
    return features.cpu(), logits.cpu()


def feature_columns(width, mode):
    hidden = width // 8
    if hidden * 8 != width:
        raise ValueError("invalid feature width")
    if mode == "answer":
        return list(range(2 * hidden))
    if mode == "last":
        return list(range(hidden, 2 * hidden)) + list(range(5 * hidden, 8 * hidden))
    if mode == "all":
        return list(range(width))
    raise ValueError("unknown feature mode")


class FrozenBevReadout:
    def __init__(self, payload, decider, device):
        import torch
        self.payload, self.decider, self.device = payload, decider, device
        self.columns = payload["columns"]
        self.mean, self.scale = payload["mean"], payload["scale"]
        self.layers = payload["layers"]
        self.temperature = float(payload.get("temperature", 1.0))
        self.permit_threshold = float(payload.get("permit_threshold", 0.0))
        self.block_threshold = float(payload.get("block_threshold", 0.0))
        if not 0 < self.temperature or not 0 <= self.permit_threshold <= 1 or not 0 <= self.block_threshold <= 1:
            raise ValueError("invalid decision thresholds")
        self.decider.network.eval().requires_grad_(False).to(device)
        self.decider.device = torch.device(device)

    def predict_features(self, feature):
        import torch
        with torch.inference_mode():
            x = (feature[self.columns].float() - self.mean) / self.scale
            for index, layer in enumerate(self.layers):
                x = torch.nn.functional.linear(x, layer["weight"], layer["bias"])
                if index < len(self.layers) - 1:
                    x = torch.nn.functional.gelu(x)
            scores = torch.softmax(x / self.temperature, dim=-1)
        selected = CHOICES[int(scores.argmax())]
        if selected == "PERMIT" and float(scores[0]) < self.permit_threshold:
            selected = "REVIEW"
        if selected == "BLOCK" and float(scores[1]) < self.block_threshold:
            selected = "REVIEW"
        return selected, {key: float(scores[i]) for i, key in enumerate(CHOICES)}

    def assess_scene(self, scene):
        if self.payload.get("input_kind") != "states":
            raise ValueError("readout was not trained for direct scenes")
        started = time.perf_counter()
        encoded = encode_scene(self.decider, scene)
        features, _ = extract_features(self.decider, encoded, self.device)
        disposition, scores = self.predict_features(features)
        return {"schema": bev.ASSESSMENT_SCHEMA, "status": "assessed",
                "disposition": disposition, "selected": disposition, "scores": scores,
                "evidence": {"profile": FEATURE_PROFILE,
                             "base_revision": bev.MODEL_REVISION, "effective_device": self.device,
                             "latency_ms": int((time.perf_counter() - started) * 1000)}}

    def assess_request(self, request, input_profile="operation_facts_v3"):
        from .sequence_candidate import encode_candidate_input
        if self.payload.get("input_kind") != "requests" or self.payload.get("input_profile") != input_profile:
            raise ValueError("readout was not trained for this request projection")
        started = time.perf_counter()
        encoded = encode_candidate_input(request, self.decider, input_profile=input_profile)
        features, _ = extract_features(self.decider, encoded["encoded"], self.device)
        disposition, scores = self.predict_features(features)
        return {"schema": bev.ASSESSMENT_SCHEMA, "status": "assessed", "selected": disposition,
                "disposition": disposition, "scores": scores,
                "evidence": {"request": encoded["clean_request"],
                             "request_sha256": encoded["request_sha256"],
                             "profile": FEATURE_PROFILE, "base_revision": bev.MODEL_REVISION,
                             "effective_device": self.device,
                             "latency_ms": int((time.perf_counter() - started) * 1000)}}


def load_readout(checkpoint_path, expected_sha256, device="cuda"):
    import torch
    path = Path(checkpoint_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError("readout hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"schema": SCHEMA, "feature_profile": FEATURE_PROFILE,
                "base_revision": bev.MODEL_REVISION, "base_weight_sha256": bev.MODEL_SHA256,
                "classes": list(CHOICES)}
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("readout provenance mismatch")
    if payload.get("feature_source_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("readout feature source changed")
    if device not in {"cpu", "cuda"}:
        raise ValueError("unsupported device")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return FrozenBevReadout(payload, bev._load_real_decider(), device)
