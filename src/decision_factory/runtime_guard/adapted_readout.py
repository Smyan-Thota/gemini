"""Limited upper-layer adapters on the pinned Bev authorization readout."""
from __future__ import annotations

import hashlib
from pathlib import Path

from . import bev, frozen_readout

SCHEMA = "bev_upper_adapter_readout/1"


def adapter_digest(state):
    from .sequence_candidate import _state_mapping_digest
    return _state_mapping_digest(state)


def add_upper_adapter(decider):
    """Freeze the pinned backbone and add rank-eight Q/V adapters to four layers."""
    import peft
    count = int(decider.network.backbone.config.num_hidden_layers)
    indices = list(range(max(0, count - 4), count))
    target = r".*layers\.(" + "|".join(str(index) for index in indices) + r")\.self_attn\.(q_proj|v_proj)"
    config = peft.LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                            target_modules=target, bias="none")
    decider.network.requires_grad_(False)
    decider.network.backbone = peft.get_peft_model(decider.network.backbone, config)
    decider.network.eval()
    return {"rank": 8, "alpha": 16, "layers": indices, "target_regex": target}


def feature_forward(decider, encoded, device="cuda", training=False):
    """Identical pooling/masks to frozen features, with optional adapter gradients."""
    import torch
    batch = {key: value.to(device) for key, value in decider.encoder.collate([encoded]).items()}
    with torch.set_grad_enabled(training):
        out = decider.network.backbone(input_ids=batch["input_ids"],
            position_ids=batch["position_ids"], attention_mask=batch["attention_mask"],
            use_cache=False, output_hidden_states=True)
        middle = out.hidden_states[len(out.hidden_states) // 2].float()
        last = out.last_hidden_state.float()
        choice = batch["choice_read_idx"][0]
        answer = int(batch["answer_end_idx"][0])
        return torch.cat([middle[0, answer], last[0, answer],
                          middle[0, choice].flatten(), last[0, choice].flatten()])


def load_adapted_readout(checkpoint_path, expected_sha256, device="cuda"):
    import peft
    import torch
    path = Path(checkpoint_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("adapted readout hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"schema": SCHEMA, "feature_profile": frozen_readout.FEATURE_PROFILE,
                "base_revision": bev.MODEL_REVISION, "base_weight_sha256": bev.MODEL_SHA256,
                "classes": list(frozen_readout.CHOICES),
                "feature_source_sha256": hashlib.sha256(Path(frozen_readout.__file__).read_bytes()).hexdigest(),
                "adaptation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if any(payload.get(key) != value for key, value in required.items()):
        raise ValueError("adapted readout provenance mismatch")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    decider = bev._load_real_decider()
    bev._cached_decider = None  # Never contaminate a future original-Bev load.
    metadata = add_upper_adapter(decider)
    if metadata != payload["adapter_config"]:
        raise ValueError("adapter configuration mismatch")
    expected_keys = set(peft.get_peft_model_state_dict(decider.network.backbone))
    if set(payload["adapter_state"]) != expected_keys:
        raise ValueError("adapter parameter coverage mismatch")
    if adapter_digest(payload["adapter_state"]) != payload["adapter_state_sha256"]:
        raise ValueError("adapter parameter digest mismatch")
    result = peft.set_peft_model_state_dict(decider.network.backbone, payload["adapter_state"])
    if result.unexpected_keys:
        raise ValueError("unexpected adapter parameter")
    return frozen_readout.FrozenBevReadout(payload, decider, device)
