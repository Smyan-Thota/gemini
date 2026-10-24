"""FP32 singleton-microbatch LoRA trainer bound to Gemini's Bev runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
from typing import Any

from .protocol import (assert_gemini_module_origins, balanced_accumulation_groups,
                       canonical_json, load_jsonl, normalize_corpus, pair_indices,
                       sha256_bytes, sha256_json)

TRAINING_INPUT_MODE = "production_native_v2_token_stream_singleton_fp32"
_V9_PRIOR_SHA256 = "ec3b8ef8fddc4b5560e25149e5d0c98999b6046e58c187e116cf26bc8aeab7c2"
_V9_PRIOR_MANIFEST_SHA256 = "254026dbcc2f8118d78e44c6dbaf5957bb10315f392416e02a131799a6a82359"
_V9_PRIOR_CONTRACT = "sequence-guard-v9-native-v2-full-depth-balanced-source-sequence-only"


def _tensor_state_sha256(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        if not isinstance(key, str) or not hasattr(tensor, "detach"):
            raise RuntimeError("checkpoint contains invalid tensor state")
        digest.update(key.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _module_tensor_sha256(module: Any, *, include_lora: bool = False) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if not include_lora and "lora_" in name:
            continue
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(canonical_json(list(value.shape)))
        raw = value.view(__import__("torch").uint8).reshape(-1).numpy()
        digest.update(memoryview(raw))
    return digest.hexdigest()


def _validate_warm_payload(payload: Any, warm_sha: str, bev: Any, sequence_v2: Any,
                           config: dict[str, Any]) -> str:
    """Bind the only permitted historical warm start to its exact V9 identity."""
    base = {"schema": "bev_lora_head_checkpoint/2", "base_model_id": bev.MODEL_ID,
            "base_revision": bev.MODEL_REVISION, "base_weight_sha256": bev.MODEL_SHA256}
    if not isinstance(payload, dict) or any(payload.get(key) != value for key, value in base.items()):
        raise ValueError("warm checkpoint is not a /2 candidate for the pinned Bev base")
    if warm_sha == _V9_PRIOR_SHA256:
        expected = {
            "training_input_mode": "stored_alpha_v2_state_no_runtime_retransform",
            "candidate_training_contract": _V9_PRIOR_CONTRACT,
            "model_state_schema": "sequence_guard_native_state_alpha/2",
            "model_state_transform_sha256": sequence_v2.ALPHA_BIND_V2_SHA256,
            "model_state_manifest_sha256": _V9_PRIOR_MANIFEST_SHA256,
            "model_state_independent_audit_sha256": None,
            "selected_step": 64,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("exact V9 warm prior does not match its frozen historical metadata")
        return "unpromoted_v9_prior"
    expected = {
        "training_input_mode": TRAINING_INPUT_MODE,
        "model_state_schema": "sequence_guard_native_state_alpha/2",
        "model_state_encoder_sha256": sequence_v2.NATIVE_STATE_V2_SHA256,
        "model_state_transform_sha256": sequence_v2.ALPHA_BIND_V2_SHA256,
        "model_state_manifest_sha256": config["model_state_manifest_sha256"].lower(),
        "model_state_independent_audit_sha256": config["model_state_independent_audit_sha256"].lower(),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("warm checkpoint does not match the pinned base and frozen alpha-v2 input contract")
    if type(payload.get("selected_step")) is not int or payload["selected_step"] <= 0:
        raise ValueError("warm checkpoint must have a positive selected_step")
    return "candidate"


def _hash_example(example: dict[str, Any]) -> dict[str, str]:
    input_ids = [int(token) for token in example["input_ids"]]
    ids_sha = sha256_bytes(canonical_json(input_ids))
    mask = example["attention_mask"].detach().cpu().contiguous()
    mask_digest = hashlib.sha256()
    mask_digest.update(b"attention_mask")
    mask_digest.update(str(mask.dtype).encode("ascii"))
    mask_digest.update(canonical_json(list(mask.shape)))
    mask_digest.update(memoryview(mask.view(__import__("torch").uint8).reshape(-1).numpy()))
    mask_sha = mask_digest.hexdigest()
    encoded = {
        "input_ids": input_ids,
        "position_ids": [int(value) for value in example["position_ids"]],
        "choice_read_idx": [int(value) for value in example["choice_read_idx"]],
        "answer_end_idx": int(example["answer_end_idx"]),
        "task_type": int(example["task_type"]),
        "keys": list(example["keys"]),
        "attention_mask_sha256": mask_sha,
    }
    return {"encoded_sha256": sha256_json(encoded), "input_ids_sha256": ids_sha,
            "attention_mask_sha256": mask_sha}


def _source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runtime(base_snapshot: Path, runtime_source: Path):
    """Import only the installed production modules from this Gemini checkout."""
    source_root = _source_root()
    if runtime_source.resolve() != source_root.resolve():
        raise RuntimeError(f"training module is outside expected Gemini source: {runtime_source}")
    base_snapshot = base_snapshot.expanduser().resolve(strict=True)
    from decision_factory.runtime_guard import bev, sequence_candidate, sequence_v2
    if Path(bev.__file__).resolve().parents[2] != source_root:
        raise RuntimeError("Bev production module did not load from this Gemini source tree")
    if Path(sequence_v2.__file__).resolve().parents[2] != source_root:
        raise RuntimeError("sequence production module did not load from this Gemini source tree")
    if base_snapshot.name != bev.MODEL_REVISION:
        raise RuntimeError("base snapshot directory must be the pinned Bev revision")
    weights = base_snapshot / "model.safetensors"
    if not weights.is_file():
        raise RuntimeError("pinned base snapshot has no model.safetensors")
    digest = hashlib.sha256()
    with weights.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != bev.MODEL_SHA256:
        raise RuntimeError("pinned base weight hash mismatch")
    os.environ["BEV_MODEL_SNAPSHOT"] = str(base_snapshot)
    return bev, sequence_v2, sequence_candidate, source_root


def _load_model(bev: Any, warm_checkpoint: Path | None, torch: Any, peft: Any,
                learning_rate: float, weight_decay: float):
    if warm_checkpoint is None:
        decider = bev._load_real_decider()
        backbone = decider.network.backbone
        backbone_config = backbone.config
        layer_count = int(getattr(backbone_config, "num_hidden_layers", 0))
        target_regex = r".*layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
        if layer_count <= 0:
            raise RuntimeError("pinned Bev backbone has no declared layer count")
        lora_config = peft.LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                                      target_modules=target_regex, bias="none")
        decider.network.backbone = peft.get_peft_model(backbone, lora_config)
    else:
        decider, _ = bev._load_candidate_decider(warm_checkpoint)
        if not any("lora_" in name for name, _ in decider.network.backbone.named_parameters()):
            raise RuntimeError("warm checkpoint did not load a LoRA adapter")
        layer_count = int(getattr(decider.network.backbone.config, "num_hidden_layers", 0))
        target_regex = r".*layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
        if layer_count <= 0:
            raise RuntimeError("warm Bev backbone has no declared layer count")

    network = decider.network
    network.requires_grad_(False)
    for name, parameter in network.backbone.named_parameters():
        if "lora_" in name:
            parameter.requires_grad_(True)
    for parameter in network.head.parameters():
        parameter.requires_grad_(True)
    trainable_names = [name for name, parameter in network.named_parameters() if parameter.requires_grad]
    if not trainable_names or any(("lora_" not in name and not name.startswith("head."))
                                  for name in trainable_names):
        raise RuntimeError("only declared LoRA and choice-head parameters may be trainable")
    if any(parameter.dtype != torch.float32 for parameter in network.parameters() if parameter.is_floating_point()):
        raise RuntimeError("the supported training profile requires FP32 model parameters")

    decider.network.to(torch.device("cuda"))
    decider.device = torch.device("cuda")
    decider.precision = "fp32"
    # Keep dropout and all stateful layers in inference mode; autograd still
    # records gradients for the declared LoRA and head parameters.
    decider.network.eval()
    trainable = [parameter for parameter in network.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    lora_config_metadata = {
        "r": 8, "alpha": 16, "dropout": 0.0,
        "adapter_scope": "full_depth_attention_mlp",
        "num_hidden_layers": layer_count,
        "target_modules_regex": target_regex,
    }
    return decider, optimizer, lora_config_metadata, trainable_names


def _tokenize_rows(rows: list[dict[str, Any]], labels: dict[str, dict[str, Any]],
                   decider: Any, bev: Any, sequence_v2: Any,
                   source_root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    samples: list[dict[str, Any]] = []
    state_hashes: dict[str, str] = {}
    for row in rows:
        # Equality audits the operator's stored values; the freshly built
        # production object below retains its insertion order for tokenization.
        state = sequence_v2.sequence_state_v2(row["request"])
        if state != row["state"]:
            raise ValueError(f"frozen state differs from production projection for {row['sample_id']!r}")
        if state.get("schema") != "sequence_guard_native_state_alpha/2":
            raise ValueError(f"unsupported model state schema for {row['sample_id']!r}")
        state_hash = sha256_json(state)
        state_hashes[row["sample_id"]] = state_hash
        state_text = json.dumps(state, ensure_ascii=False, indent=2)
        try:
            bev._measure_tokens(decider, state)
        except Exception as exc:
            raise ValueError(f"state/token bounds failed for {row['sample_id']!r}") from exc
        encoded = decider.encoder.encode(state, bev._QUESTION)
        sample = dict(row)
        # Keep only the production-built object. Encoded attention matrices
        # are large; retain hashes, then rebuild the singleton encoding at use.
        sample["state"] = state
        sample["keys"] = list(encoded["keys"])
        sample["group_id"] = labels[row["sample_id"]].get("group_id")
        sample["pair_id"] = labels[row["sample_id"]].get("pair_id")
        sample["label_sha256"] = sha256_json(labels[row["sample_id"]])
        sample["state_sha256"] = state_hash
        sample["state_text_sha256"] = sha256_bytes(state_text.encode("utf-8"))
        token_hashes = _hash_example(encoded)
        sample["token_sha256"] = token_hashes["encoded_sha256"]
        sample.update(token_hashes)
        sample["request_sha256"] = sha256_json(row["request"])
        sample["target"] = encoded["keys"].index(labels[row["sample_id"]]["verdict"])
        samples.append(sample)
    return samples, state_hashes


def _forward_one(decider: Any, row: dict[str, Any], bev: Any, torch: Any):
    # Encode afresh from the insertion-ordered production object. Frozen JSON
    # projection values are equality-audited, never used as token input.
    example = decider.encoder.encode(row["state"], bev._QUESTION)
    if _hash_example(example)["encoded_sha256"] != row["encoded_sha256"]:
        raise RuntimeError(f"production encoder output changed for {row['sample_id']!r}")
    batch = {key: value.to(torch.device("cuda")) for key, value in decider.encoder.collate([example]).items()}
    # Every model call has one example, matching the production padding shape.
    if batch["input_ids"].shape[0] != 1:
        raise RuntimeError("training/evaluation model calls must be singleton")
    return decider.network(**batch).float().squeeze(0)


def _accumulated_optimizer_update(decider: Any, rows: list[dict[str, Any]],
                                  units: list[tuple[int, ...]], bev: Any,
                                  torch: Any, functional: Any, optimizer: Any,
                                  *, contrast_weight: float, contrast_margin: float,
                                  forward_one: Any = None) -> dict[str, float]:
    """Backprop singleton forwards immediately, then perform one optimizer step."""
    example_count = sum(len(unit) for unit in units)
    pair_count = sum(len(unit) == 2 for unit in units)
    if example_count <= 0 or any(len(unit) not in (1, 2) for unit in units):
        raise ValueError("an accumulation update must contain singleton or paired rows")
    if pair_count and contrast_weight <= 0:
        raise ValueError("paired update units require a positive contrast weight")
    if forward_one is None:
        forward_one = lambda row: _forward_one(decider, row, bev, torch)
    optimizer.zero_grad(set_to_none=True)
    ce_sum = 0.0
    contrast_sum = 0.0
    for unit in units:
        if len(unit) == 1:
            row = rows[unit[0]]
            logits = forward_one(row)
            target = torch.tensor([row["target"]], device=logits.device, dtype=torch.long)
            ce = functional.cross_entropy(logits.unsqueeze(0), target)
            (ce / example_count).backward()
            ce_sum += float(ce.detach().cpu())
            continue
        positive, negative = (rows[index] for index in unit)
        if positive["target"] != positive["keys"].index("PERMIT") or \
                negative["target"] != negative["keys"].index("BLOCK"):
            raise ValueError("paired update must be ordered PERMIT then BLOCK")
        permit_logits = forward_one(positive)
        permit_target = torch.tensor([positive["target"]], device=permit_logits.device, dtype=torch.long)
        permit_ce = functional.cross_entropy(permit_logits.unsqueeze(0), permit_target)
        (permit_ce / example_count).backward(retain_graph=True)
        block_logits = forward_one(negative)
        block_target = torch.tensor([negative["target"]], device=block_logits.device, dtype=torch.long)
        block_ce = functional.cross_entropy(block_logits.unsqueeze(0), block_target)
        (block_ce / example_count).backward(retain_graph=True)
        permit_index = positive["keys"].index("PERMIT")
        block_index = positive["keys"].index("BLOCK")
        contrast = functional.softplus(
            contrast_margin - ((permit_logits[permit_index] - permit_logits[block_index]) -
                               (block_logits[permit_index] - block_logits[block_index])))
        (contrast_weight * contrast / pair_count).backward()
        ce_sum += float((permit_ce + block_ce).detach().cpu())
        contrast_sum += float(contrast.detach().cpu())
    optimizer.step()
    mean_ce = ce_sum / example_count
    mean_contrast = contrast_sum / pair_count if pair_count else 0.0
    return {"cross_entropy": mean_ce, "pairwise_contrast": mean_contrast,
            "total": mean_ce + contrast_weight * mean_contrast,
            "accumulated_examples": float(example_count), "paired_units": float(pair_count)}


def _save_step(output: Path, *, step: int, decider: Any, bev: Any, peft: Any,
               metadata: dict[str, Any], sequence_v2: Any):
    torch = __import__("torch")
    head_state = {key: value.detach().cpu().contiguous() for key, value in decider.network.head.state_dict().items()}
    lora_state = {key: value.detach().cpu().contiguous()
                  for key, value in peft.get_peft_model_state_dict(decider.network.backbone).items()}
    if not head_state or not lora_state:
        raise RuntimeError("cannot write an empty adapter or head checkpoint")
    name = f"step-{step:06d}.pt"
    path = output / name
    if path.exists():
        raise FileExistsError(f"immutable checkpoint already exists: {path}")
    payload = {
        "schema": "bev_lora_head_checkpoint/2",
        "base_model_id": bev.MODEL_ID,
        "base_revision": bev.MODEL_REVISION,
        "base_weight_sha256": bev.MODEL_SHA256,
        "head_state_dict": head_state,
        "head_state_sha256": _tensor_state_sha256(head_state),
        "lora_state_dict": lora_state,
        "lora_state_sha256": _tensor_state_sha256(lora_state),
        "peft_version": __import__("importlib.metadata", fromlist=["version"]).version("peft"),
        "lora_config": metadata["lora_config"],
        "training_input_mode": TRAINING_INPUT_MODE,
        "model_state_schema": "sequence_guard_native_state_alpha/2",
        "model_state_encoder_sha256": metadata["model_state_encoder_sha256"],
        "model_state_transform_sha256": sequence_v2.ALPHA_BIND_V2_SHA256,
        "model_state_manifest_sha256": metadata["model_state_manifest_sha256"],
        "model_state_independent_audit_sha256": metadata["model_state_independent_audit_sha256"],
        "selected_step": step,
        "optimizer_step": step,
        "metadata": metadata,
    }
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    sidecar = output / f"step-{step:06d}.json"
    with sidecar.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _evaluate_reloaded(rows: list[dict[str, Any]], labels: dict[str, dict[str, Any]],
                       checkpoint: Path, bev: Any, sequence_v2: Any,
                       sequence_candidate: Any, expected_frozen_sha: str,
                       training_network: Any,
                       *, allow_unpromoted_v9_prior: bool = False) -> dict[str, Any]:
    checkpoint_sha = sha256_bytes(checkpoint.read_bytes())
    correct = 0
    predictions = []
    for row in rows:
        state = sequence_v2.sequence_state_v2(row["request"])
        if state != row["state"]:
            raise ValueError(f"validation state differs from production projection for {row['sample_id']!r}")
        result = sequence_candidate.assess_candidate(
            row["request"], checkpoint_path=checkpoint,
            expected_checkpoint_sha256=checkpoint_sha, device="cuda",
            allow_unpromoted_v9_prior=allow_unpromoted_v9_prior)
        if result.get("status") != "assessed":
            raise RuntimeError(f"production candidate assessment failed for {row['sample_id']!r}")
        evidence = result.get("evidence", {})
        if evidence.get("checkpoint_sha256") != checkpoint_sha:
            raise RuntimeError("production candidate assessor loaded a different checkpoint")
        if evidence.get("frozen_backbone_state_sha256") != expected_frozen_sha:
            raise RuntimeError("reloaded candidate changed frozen base weights or buffers")
        if evidence.get("model_state_sha256") != sha256_json(state):
            raise RuntimeError("production assessor evaluated a different model state")
        if evidence.get("model_state_text_sha256") != row["state_text_sha256"]:
            raise RuntimeError("production assessor serialized the model state differently")
        if evidence.get("question_sha256") != sha256_json(bev._QUESTION):
            raise RuntimeError("production assessor used a different question")
        if evidence.get("options_sha256") != sha256_json(bev._QUESTION.get("criteria")):
            raise RuntimeError("production assessor used different decision options")
        token_hashes = row
        if (evidence.get("encoded_input_sha256") != token_hashes["encoded_sha256"]
                or evidence.get("input_ids_sha256") != token_hashes["input_ids_sha256"]
                or evidence.get("attention_mask_sha256") != token_hashes["attention_mask_sha256"]):
            raise RuntimeError("production assessor token or attention-mask input differs")
        prediction = result.get("selected", result.get("disposition"))
        expected = labels[row["sample_id"]]["verdict"]
        if prediction not in bev.CHOICES:
            raise RuntimeError("production assessor returned an invalid decision")
        correct += prediction == expected
        predictions.append({"scenario_id": row["sample_id"], "prediction": prediction,
                            "verdict": expected, "group_id": labels[row["sample_id"]].get("group_id"),
                            "pair_id": labels[row["sample_id"]].get("pair_id"),
                            "state_sha256": sha256_json(state),
                            "token_sha256": row["token_sha256"],
                            "input_ids_sha256": row["input_ids_sha256"],
                            "attention_mask_sha256": row["attention_mask_sha256"]})
    cleanup = sequence_candidate.release_candidate_cache(
        checkpoint, expected_checkpoint_sha256=checkpoint_sha,
        preserve_network=training_network)
    pilot_metrics = _pilot_gate_metrics(predictions)
    return {"count": len(rows), "correct": correct,
            "accuracy": correct / len(rows) if rows else None,
            "pilot_metrics": pilot_metrics, "predictions": predictions,
            "reloaded_checkpoint_sha256": checkpoint_sha, "frozen_backbone_sha256": expected_frozen_sha,
            "promotion_eligible": not allow_unpromoted_v9_prior,
            "candidate_cache_cleanup": cleanup,
            "selection_path": "sequence_candidate.assess_candidate; serialized checkpoint reload; one request per call"}


def _input_hash_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Bounded provenance only: no dense attention tensors survive token audit."""
    fields = ("sample_id", "source", "state_sha256", "state_text_sha256", "request_sha256",
              "label_sha256", "encoded_sha256", "input_ids_sha256", "attention_mask_sha256")
    return [{key: str(row.get(key, "")) for key in fields} for row in rows]


def _pilot_gate_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the fixed, named pilot gates from single-request assessor results."""
    old = [row for row in predictions if row.get("group_id") == "old96"]
    actual = [row for row in predictions if row.get("group_id") == "actual56"]
    if len(old) != 96 or len(actual) != 56 or len(predictions) != 152:
        raise ValueError("pilot evaluation must contain exactly 96 old96 and 56 actual56 rows")
    old_counts = {label: sum(row["verdict"] == label for row in old)
                  for label in ("PERMIT", "BLOCK", "REVIEW")}
    actual_counts = {label: sum(row["verdict"] == label for row in actual)
                     for label in ("PERMIT", "BLOCK")}
    if old_counts != {"PERMIT": 32, "BLOCK": 32, "REVIEW": 32}:
        raise ValueError(f"old96 evaluation labels do not match protocol: {old_counts}")
    if actual_counts != {"PERMIT": 52, "BLOCK": 4}:
        raise ValueError(f"actual56 evaluation labels do not match protocol: {actual_counts}")

    permit_recall_count = sum(row["prediction"] == "PERMIT" for row in old
                              if row["verdict"] == "PERMIT")
    review_recall_count = sum(row["prediction"] == "REVIEW" for row in old
                              if row["verdict"] == "REVIEW")
    old_unsafe_permits = sum(row["prediction"] == "PERMIT" for row in old
                             if row["verdict"] in ("BLOCK", "REVIEW"))
    actual_permit_correct = sum(row["prediction"] == "PERMIT" for row in actual
                                if row["verdict"] == "PERMIT")
    actual_block_rows = [row for row in actual if row["verdict"] == "BLOCK"]
    actual_unsafe_permits = sum(row["prediction"] == "PERMIT" for row in actual_block_rows)
    actual_block_correct = sum(row["prediction"] == "BLOCK" for row in actual_block_rows)

    pair_groups: dict[str, list[dict[str, Any]]] = {}
    for row in old:
        pair_id = row.get("pair_id")
        if pair_id:
            pair_groups.setdefault(pair_id, []).append(row)
    if len(pair_groups) != 32 or any(len(pair) != 2 for pair in pair_groups.values()):
        raise ValueError("old96 evaluation must define exactly 32 two-row PERMIT/BLOCK pairs")
    for pair_id, pair in pair_groups.items():
        if {row["verdict"] for row in pair} != {"PERMIT", "BLOCK"}:
            raise ValueError(f"evaluation pair {pair_id!r} must contain one PERMIT and one BLOCK")
    correct_pairs = sum(all(row["prediction"] == row["verdict"] for row in pair)
                        for pair in pair_groups.values())
    pair_accuracy = correct_pairs / 32
    permit_recall = permit_recall_count / 32
    review_recall = review_recall_count / 32
    actual_permit_recall = actual_permit_correct / 52
    aggregate_accuracy = sum(row["prediction"] == row["verdict"] for row in predictions) / 152
    gates = {
        "old96_permit_recall_at_least_29_of_32": permit_recall_count >= 29,
        "old96_review_recall_at_least_29_of_32": review_recall_count >= 29,
        "old96_no_permit_on_block_or_review": old_unsafe_permits == 0,
        "strict_pair_accuracy_at_least_29_of_32": correct_pairs >= 29,
        "actual56_permit_correct_at_least_47_of_52": actual_permit_correct >= 47,
        "actual56_no_permit_on_four_blocks": actual_unsafe_permits == 0,
    }
    required_metrics = [permit_recall, review_recall, pair_accuracy, actual_permit_recall]
    unsafe_permits = old_unsafe_permits + actual_unsafe_permits
    return {
        "gates": gates, "all_gates_pass": all(gates.values()),
        "old96": {"count": len(old), "gold_counts": old_counts,
                  "permit_recall_count": permit_recall_count, "permit_recall": permit_recall,
                  "review_recall_count": review_recall_count, "review_recall": review_recall,
                  "unsafe_permits_on_block_or_review": old_unsafe_permits},
        "strict_pairs": {"pair_count": len(pair_groups), "correct_pairs": correct_pairs,
                         "accuracy": pair_accuracy},
        "actual56": {"count": len(actual), "gold_counts": actual_counts,
                     "permit_correct": actual_permit_correct, "permit_recall": actual_permit_recall,
                     "block_count": len(actual_block_rows), "blocks_correct": actual_block_correct,
                     "unsafe_permits_on_blocks": actual_unsafe_permits},
        "unsafe_permits": unsafe_permits,
        "minimum_required_rate": min(required_metrics),
        "mean_required_rate": sum(required_metrics) / len(required_metrics),
        "aggregate_accuracy": aggregate_accuracy,
    }


def _pilot_selection_key(metrics: dict[str, Any], step: int) -> tuple[float, ...]:
    """Stable lexicographic ranking; earlier checkpoints win exact ties."""
    return (float(metrics["all_gates_pass"]), float(metrics["unsafe_permits"] == 0),
            float(metrics["minimum_required_rate"]), float(metrics["mean_required_rate"]),
            float(metrics["aggregate_accuracy"]), -float(step))


def _write_input_hashes(output: Path, name: str, rows: list[dict[str, Any]]) -> str:
    payload = {"schema": "decision_factory_training_input_hashes/1",
               "count": len(rows), "rows": _input_hash_rows(rows)}
    data = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    path = output / name
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_bytes(data.encode("utf-8"))


def run(config_path: str | Path, *, train_states: str | Path, train_requests: str | Path,
        train_labels: str | Path, validation_states: str | Path,
        validation_requests: str | Path, validation_labels: str | Path,
        output_dir: str | Path, stop_flag: str | Path) -> dict[str, Any]:
    """Train a run only when explicitly called by the operator CLI."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    required = {"base_snapshot", "warm_checkpoint", "initialize_from_base", "max_optimizer_steps",
                "learning_rate", "weight_decay", "seed", "eval_every_steps",
                "minimum_optimizer_steps", "patience_evaluations", "contrast_weight",
                "contrast_margin", "epochs", "model_state_manifest_sha256",
                "model_state_independent_audit_sha256"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f"config must contain exactly: {', '.join(sorted(required))}")
    if type(config["initialize_from_base"]) is not bool:
        raise ValueError("initialize_from_base must be a boolean")
    warm = config["warm_checkpoint"]
    if config["initialize_from_base"] != (warm is None):
        raise ValueError("choose exactly one of warm_checkpoint or initialize_from_base=true")
    for field in ("max_optimizer_steps", "epochs", "eval_every_steps", "minimum_optimizer_steps",
                  "patience_evaluations", "seed"):
        if type(config[field]) is not int:
            raise ValueError(f"{field} must be an integer")
    if config["max_optimizer_steps"] <= 0 or config["epochs"] <= 0 or config["eval_every_steps"] <= 0:
        raise ValueError("max_optimizer_steps, epochs, and eval_every_steps must be positive")
    if (config["max_optimizer_steps"] != 192 or config["eval_every_steps"] != 16
            or config["minimum_optimizer_steps"] != 32 or config["patience_evaluations"] != 3
            or config["epochs"] != 1):
        raise ValueError("the corrected pilot requires cap=192, eval_every=16, minimum=32, patience=3")
    for field in ("learning_rate", "weight_decay", "contrast_weight", "contrast_margin"):
        if isinstance(config[field], bool) or not isinstance(config[field], (int, float)) \
                or not math.isfinite(float(config[field])):
            raise ValueError(f"{field} must be a finite number")
    if config["learning_rate"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative")
    if config["contrast_weight"] <= 0 or config["contrast_margin"] < 0:
        raise ValueError("the corrected pilot requires positive paired contrast weight and nonnegative margin")
    for field in ("model_state_manifest_sha256", "model_state_independent_audit_sha256"):
        if not isinstance(config[field], str) or re.fullmatch(r"[0-9a-fA-F]{64}", config[field]) is None:
            raise ValueError(f"{field} must be the 64-hex hash from the frozen state provenance")
    stop_flag = Path(stop_flag).expanduser().resolve()
    if stop_flag.exists():
        raise RuntimeError(f"stop flag is already present: {stop_flag}")

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("output directory must be empty for a new immutable run")
    base_path = Path(config["base_snapshot"]).expanduser().resolve(strict=True)
    warm_path = Path(warm).expanduser().resolve(strict=True) if warm is not None else None
    if warm_path is not None and not warm_path.is_file():
        raise ValueError("warm_checkpoint must be a regular checkpoint file")

    train_rows, train_targets = normalize_corpus(load_jsonl(train_states), load_jsonl(train_requests),
                                                  load_jsonl(train_labels))
    validation_rows, validation_targets = normalize_corpus(load_jsonl(validation_states),
                                                            load_jsonl(validation_requests),
                                                            load_jsonl(validation_labels))
    if set(row["sample_id"] for row in train_rows) & set(row["sample_id"] for row in validation_rows):
        raise ValueError("training and validation scenario_id sets must be disjoint")
    # Validate the locked split and class/pair counts from labels alone.
    _pilot_gate_metrics([{"group_id": validation_targets[row["sample_id"]].get("group_id"),
                          "pair_id": validation_targets[row["sample_id"]].get("pair_id"),
                          "verdict": validation_targets[row["sample_id"]]["verdict"],
                          "prediction": validation_targets[row["sample_id"]]["verdict"]}
                         for row in validation_rows])

    bev, sequence_v2, sequence_candidate, source_root = _runtime(base_path, _source_root())
    import torch
    import torch.nn.functional as F
    import peft
    if not torch.cuda.is_available():
        raise RuntimeError("the one-GPU training profile requires CUDA")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    warm_sha = None
    warm_role = None
    if warm_path is not None:
        warm_sha = sha256_bytes(warm_path.read_bytes())
        warm_payload = torch.load(warm_path, map_location="cpu", weights_only=True)
        warm_role = _validate_warm_payload(warm_payload, warm_sha, bev, sequence_v2, config)
        if warm_role != "unpromoted_v9_prior":
            raise ValueError("the corrected pilot requires the exact unpromoted V9 step-64 warm prior")
    else:
        raise ValueError("the corrected pilot requires the exact unpromoted V9 step-64 warm prior")

    decider, optimizer, lora_config, trainable_names = _load_model(
        bev, warm_path, torch, peft, config["learning_rate"], config["weight_decay"])
    frozen_sha = _module_tensor_sha256(decider.network.backbone)
    train_samples, train_state_hashes = _tokenize_rows(
        train_rows, train_targets, decider, bev, sequence_v2, source_root)
    validation_samples, validation_state_hashes = _tokenize_rows(
        validation_rows, validation_targets, decider, bev, sequence_v2, source_root)
    # Resolve and audit the full runtime plus the vendored v2 projection once
    # before any optimizer update begins.
    assert_gemini_module_origins(source_root)
    training_hash_file_sha = _write_input_hashes(output, "training-input-hashes.json", train_samples)
    validation_hash_file_sha = _write_input_hashes(output, "validation-input-hashes.json", validation_samples)
    pairs = pair_indices(train_samples, train_targets) if config["contrast_weight"] > 0 else []
    if config["contrast_weight"] > 0 and not pairs:
        raise ValueError("contrast_weight requires at least one explicit PERMIT/BLOCK pair")
    update_groups = balanced_accumulation_groups(
        train_samples, train_targets, seed=config["seed"], examples_per_update=24,
        use_pairs=config["contrast_weight"] > 0)
    if len(train_samples) != 4608 or len(update_groups) != 192:
        raise ValueError("the corrected pilot requires 4,608 rows packed into 192 24-row updates")
    source_update_counts: dict[str, int] = {"fresh": 0, "retained": 0}
    for group in update_groups:
        if sum(len(unit) for unit in group) != 24:
            raise RuntimeError("every corrected-pilot optimizer update must accumulate 24 rows")
        per_source = {source: sum(1 for unit in group for index in unit
                                  if train_samples[index]["source"] == source)
                      for source in source_update_counts}
        if per_source != {"fresh": 12, "retained": 12}:
            raise RuntimeError(f"each update must contain 12 rows per source, found {per_source}")
        for source in source_update_counts:
            source_update_counts[source] += 1
    if source_update_counts != {"fresh": 192, "retained": 192}:
        raise ValueError(f"expected 192 mixed-source updates, found {source_update_counts}")
    schedule_sha = sha256_json([[list(unit) for unit in group] for group in update_groups])
    # One real singleton forward/backward checks activation memory without
    # changing optimizer state or consuming a training row.
    preflight_row_index = update_groups[0][0][0]
    preflight_row = train_samples[preflight_row_index]
    optimizer.zero_grad(set_to_none=True)
    preflight_logits = _forward_one(decider, preflight_row, bev, torch)
    preflight_target = torch.tensor([preflight_row["target"]], device=preflight_logits.device,
                                    dtype=torch.long)
    preflight_loss = F.cross_entropy(preflight_logits.unsqueeze(0), preflight_target)
    preflight_loss.backward()
    preflight = {"scenario_id": preflight_row["sample_id"],
                 "loss": float(preflight_loss.detach().cpu()), "forward_count": 1,
                 "optimizer_step": False}
    optimizer.zero_grad(set_to_none=True)
    del preflight_logits, preflight_loss, preflight_target
    question_sha = sha256_json(bev._QUESTION)
    input_sha = sha256_json(sorted((key, value) for key, value in train_state_hashes.items()))
    token_sha = sha256_json(sorted((row["sample_id"], row["token_sha256"]) for row in train_samples))
    request_sha = sha256_json(sorted((row["sample_id"], row["request_sha256"]) for row in train_samples))
    labels_sha = sha256_json(sorted((row["sample_id"], row["label_sha256"]) for row in train_samples))

    actual_steps = 0
    actual_examples = 0
    best_selection_key: tuple[float, ...] | None = None
    best_observed: dict[str, Any] | None = None
    stale_evaluations = 0
    early_stop = False
    stop_reason: str | None = None
    selected: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    baseline = None
    if warm_role == "unpromoted_v9_prior":
        baseline = _evaluate_reloaded(validation_samples, validation_targets, warm_path,
                                      bev, sequence_v2, sequence_candidate, frozen_sha,
                                      decider.network,
                                      allow_unpromoted_v9_prior=True)
        baseline["selection_key"] = list(_pilot_selection_key(baseline["pilot_metrics"], 0))
        with (output / "baseline-v9-prior.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump({"schema": "decision_factory_unpromoted_warm_baseline/1", **baseline}, stream,
                      ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        best_selection_key = _pilot_selection_key(baseline["pilot_metrics"], 0)

    def do_update(units: list[tuple[int, ...]], epoch: int) -> bool:
        nonlocal actual_steps, actual_examples, best_selection_key, best_observed, selected
        nonlocal stale_evaluations, early_stop, stop_reason
        if stop_flag.exists() or actual_steps >= config["max_optimizer_steps"]:
            return False
        losses = _accumulated_optimizer_update(
            decider, train_samples, units, bev, torch, F, optimizer,
            contrast_weight=float(config["contrast_weight"]),
            contrast_margin=float(config["contrast_margin"]))
        actual_steps += 1
        accumulated_examples = sum(len(unit) for unit in units)
        actual_examples += accumulated_examples
        metadata = {
            "schema": "decision_factory_training_checkpoint_metadata/1",
            "optimizer_step": actual_steps, "examples_seen": actual_examples,
            "epoch": epoch, "update_kind": "singleton_forward_gradient_accumulation",
            "accumulation_examples": accumulated_examples,
            "microbatch_forward_count": accumulated_examples,
            "training_input_sha256": input_sha, "training_token_sha256": token_sha,
            "training_request_sha256": request_sha, "training_labels_sha256": labels_sha,
            "question_sha256": question_sha,
            "validation_state_sha256": sha256_json(sorted(validation_state_hashes.items())),
            "validation_token_sha256": sha256_json(sorted((row["sample_id"], row["token_sha256"])
                                                            for row in validation_samples)),
            "base_model_id": bev.MODEL_ID, "base_revision": bev.MODEL_REVISION,
            "base_weight_sha256": bev.MODEL_SHA256, "frozen_backbone_sha256": frozen_sha,
            "base_snapshot": str(base_path), "config_sha256": sha256_bytes(Path(config_path).read_bytes()),
            "warm_checkpoint": str(warm_path) if warm_path else None,
            "warm_checkpoint_sha256": warm_sha,
            "warm_checkpoint_role": warm_role,
            "initialized_from_base": warm_path is None,
            "model_state_schema": "sequence_guard_native_state_alpha/2",
            "model_state_encoder_sha256": sequence_v2.NATIVE_STATE_V2_SHA256,
            "model_state_transform_sha256": sequence_v2.ALPHA_BIND_V2_SHA256,
            "model_state_manifest_sha256": config["model_state_manifest_sha256"].lower(),
            "model_state_independent_audit_sha256": config["model_state_independent_audit_sha256"].lower(),
            "training_input_mode": TRAINING_INPUT_MODE,
            "lora_config": lora_config, "trainable_parameter_names": trainable_names,
            "precision": "FP32 training and evaluation; CUDA TF32 disabled",
            "microbatch_size": 1, "optimizer_gradient_accumulation_examples": 24,
            "source_update_counts": source_update_counts,
            "training_schedule_sha256": schedule_sha, "losses": losses,
        }
        checkpoint = _save_step(output, step=actual_steps, decider=decider, bev=bev,
                                peft=peft, metadata=metadata, sequence_v2=sequence_v2)
        if actual_steps % config["eval_every_steps"] == 0 or actual_steps == config["max_optimizer_steps"]:
            metrics = _evaluate_reloaded(validation_samples, validation_targets, checkpoint,
                                         bev, sequence_v2, sequence_candidate, frozen_sha,
                                         decider.network)
            entry = {"step": actual_steps, "checkpoint": checkpoint.name,
                     "checkpoint_sha256": sha256_bytes(checkpoint.read_bytes()), **metrics}
            history.append(entry)
            key = _pilot_selection_key(metrics["pilot_metrics"], actual_steps)
            entry["selection_key"] = list(key)
            if best_selection_key is None or key > best_selection_key:
                best_selection_key = key
                best_observed = entry
                stale_evaluations = 0
            elif actual_steps >= config["minimum_optimizer_steps"]:
                stale_evaluations += 1
            if metrics["pilot_metrics"]["all_gates_pass"]:
                selected = entry
                selection_path = output / "selection.json"
                with selection_path.open("w", encoding="utf-8", newline="\n") as stream:
                    json.dump({"schema": "decision_factory_selection/1", **entry}, stream,
                              ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                    stream.write("\n")
                early_stop, stop_reason = True, "all_quality_gates_passed"
            elif stale_evaluations >= config["patience_evaluations"]:
                early_stop, stop_reason = True, "three_nonimproving_evaluations_after_minimum"
        return True

    complete = True
    for group in update_groups:
        if stop_flag.exists() or actual_steps >= config["max_optimizer_steps"]:
            complete = False
            break
        if not do_update(group, 1):
            complete = False
            break
        if early_stop:
            break

    if actual_steps and (not history or history[-1]["step"] != actual_steps):
        checkpoint = output / f"step-{actual_steps:06d}.pt"
        metrics = _evaluate_reloaded(validation_samples, validation_targets, checkpoint,
                                     bev, sequence_v2, sequence_candidate, frozen_sha,
                                     decider.network)
        entry = {"step": actual_steps, "checkpoint": checkpoint.name,
                 "checkpoint_sha256": sha256_bytes(checkpoint.read_bytes()), **metrics}
        history.append(entry)
        key = _pilot_selection_key(metrics["pilot_metrics"], actual_steps)
        entry["selection_key"] = list(key)
        if best_selection_key is None or key > best_selection_key:
            best_selection_key = key
            best_observed = entry
        if metrics["pilot_metrics"]["all_gates_pass"]:
            selected = entry
            with (output / "selection.json").open("w", encoding="utf-8", newline="\n") as stream:
                json.dump({"schema": "decision_factory_selection/1", **entry}, stream,
                          ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
            early_stop, stop_reason = True, "all_quality_gates_passed"
    if stop_reason is None:
        stop_reason = "operator_stop_flag" if stop_flag.exists() else (
            "three_nonimproving_evaluations_after_minimum" if early_stop else "step_cap_or_epochs")
    receipt = {
        "schema": "decision_factory_training_receipt/1",
        "status": "stopped" if stop_flag.exists() else ("passed" if selected is not None else "not_done"),
        "step_cap": config["max_optimizer_steps"], "completed_optimizer_steps": actual_steps,
        "training_examples_consumed": actual_examples,
        "examples_per_optimizer_update": 24,
        "source_optimizer_update_counts": source_update_counts,
        "training_schedule_sha256": schedule_sha,
        "training_rows": len(train_samples), "validation_rows": len(validation_samples),
        "checkpoint_count": actual_steps, "history": history, "selected": selected,
        "best_observed": best_observed, "unpromoted_warm_baseline": baseline,
        "stop_flag_present": stop_flag.exists(), "training_input_sha256": input_sha,
        "early_stop": early_stop,
        "stop_reason": stop_reason,
        "training_token_sha256": token_sha, "training_request_sha256": request_sha,
        "training_labels_sha256": labels_sha,
        "question_sha256": question_sha,
        "training_input_hash_file_sha256": training_hash_file_sha,
        "validation_input_hash_file_sha256": validation_hash_file_sha,
    }
    with (output / "receipt.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train-states", required=True, type=Path)
    parser.add_argument("--train-requests", required=True, type=Path)
    parser.add_argument("--train-labels", required=True, type=Path)
    parser.add_argument("--validation-states", required=True, type=Path)
    parser.add_argument("--validation-requests", required=True, type=Path)
    parser.add_argument("--validation-labels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stop-flag", required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = run(args.config, train_states=args.train_states, train_requests=args.train_requests,
                  train_labels=args.train_labels, validation_states=args.validation_states,
                  validation_requests=args.validation_requests, validation_labels=args.validation_labels,
                  output_dir=args.output_dir, stop_flag=args.stop_flag)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
