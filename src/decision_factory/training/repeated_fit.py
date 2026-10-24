"""Bounded TRAIN-only memorization diagnostic for the frozen V3 probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

from .protocol import (assert_gemini_module_origins, canonical_json, load_jsonl,
                       normalize_corpus, sha256_bytes, sha256_json)
from . import trainer

PROFILE = trainer.INPUT_PROFILE_V3
SNAPSHOT_SCHEMA = "gemini_train_only_repeated_fit_snapshot/1"


def memorization_gate(correct: int, mean_cross_entropy: float) -> bool:
    """Return whether one TRAIN-only check satisfies the fixed fit threshold."""
    return correct == 24 and mean_cross_entropy <= 0.05


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_progress(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scores(result: dict[str, Any], choices: list[str]) -> dict[str, float]:
    raw = result.get("scores")
    if not isinstance(raw, dict) or any(key not in raw for key in choices):
        raise RuntimeError("production assessor did not return all choice scores")
    values = {key: float(raw[key]) for key in choices}
    if any(not 0.0 <= value <= 1.0 for value in values.values()):
        raise RuntimeError("production assessor scores are not probabilities")
    return values


def _assessment(rows, labels, checkpoint: Path, bev, sequence_candidate,
                frozen_sha: str, training_network, *, retain_predictions: bool = True):
    checkpoint_sha = _digest_file(checkpoint)
    correct, loss_sum, predictions = 0, 0.0, []
    for row in rows:
        result = sequence_candidate.assess_candidate(
            row["request"], checkpoint_path=checkpoint,
            expected_checkpoint_sha256=checkpoint_sha, device="cuda",
            input_profile=PROFILE)
        if result.get("status") != "assessed":
            raise RuntimeError(f"production assessor failed for TRAIN row {row['sample_id']}")
        evidence = result.get("evidence", {})
        for key, expected in {
            "checkpoint_sha256": checkpoint_sha,
            "model_state_profile": PROFILE,
            "frozen_backbone_state_sha256": frozen_sha,
            "model_state_sha256": row["state_sha256"],
            "model_state_text_sha256": row["state_text_sha256"],
            "question_sha256": row["question_sha256"],
            "options_sha256": row["options_sha256"],
            "encoded_input_sha256": row["encoded_sha256"],
            "input_ids_sha256": row["input_ids_sha256"],
            "attention_mask_sha256": row["attention_mask_sha256"],
            "inference_batch_size": 1,
            "precision": "fp32", "tf32": False,
        }.items():
            if evidence.get(key) != expected:
                raise RuntimeError(f"assessor fingerprint mismatch on {row['sample_id']}: {key}")
        truth = labels[row["sample_id"]]["verdict"]
        choices = row["keys"]
        scores = _scores(result, choices)
        probability = max(scores[truth], 1e-30)
        loss_sum -= __import__("math").log(probability)
        prediction = result.get("selected", result.get("disposition"))
        correct += prediction == truth
        if retain_predictions:
            predictions.append({"scenario_id": row["sample_id"], "prediction": prediction,
                                "verdict": truth, "scores": scores,
                                "state_sha256": row["state_sha256"],
                                "encoded_input_sha256": row["encoded_sha256"]})
    cleanup = sequence_candidate.release_candidate_cache(
        checkpoint, expected_checkpoint_sha256=checkpoint_sha,
        preserve_network=training_network)
    return {"count": len(rows), "correct": correct, "accuracy": correct / len(rows),
            "mean_cross_entropy": loss_sum / len(rows), "checkpoint_sha256": checkpoint_sha,
            "predictions": predictions, "cache_cleanup": cleanup,
            "assessment_path": "production assessor, serialized checkpoint reload, singleton CUDA FP32"}


def run(config_path: str | Path, *, request_path: str | Path, label_path: str | Path,
        manifest_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    required = {"base_snapshot", "seed", "learning_rate", "weight_decay", "contrast_weight",
                "contrast_margin", "max_optimizer_steps", "eval_every_steps",
                "model_state_manifest_sha256", "model_state_independent_audit_sha256",
                "probe_manifest_sha256", "requests_sha256", "labels_sha256"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("diagnostic config fields do not match the frozen protocol")
    if (config["max_optimizer_steps"] != 256 or config["eval_every_steps"] != 16
            or config["seed"] != 17):
        raise ValueError("repeated-fit schedule is fixed at seed 17, cap 256, eval every 16")
    request_path, label_path = Path(request_path).resolve(strict=True), Path(label_path).resolve(strict=True)
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (_digest_file(manifest_path) != config["probe_manifest_sha256"]
            or _digest_file(request_path) != config["requests_sha256"]
            or _digest_file(label_path) != config["labels_sha256"]
            or manifest.get("input_profile") != PROFILE):
        raise RuntimeError("frozen TRAIN probe hash/profile mismatch")
    labels_in = load_jsonl(label_path)
    if any(not {"scenario_id", "verdict"} <= set(item) for item in labels_in):
        raise ValueError("probe sidecar requires separate scenario_id/verdict labels")
    request_rows = load_jsonl(request_path)
    if len(request_rows) != 24 or len(labels_in) != 24:
        raise ValueError("repeated-fit diagnostic is exactly 24 frozen TRAIN examples")
    ids = [row.get("scenario_id") for row in request_rows]
    if len(set(ids)) != 24 or any(set(row) != {"scenario_id", "guard_request"} for row in request_rows):
        raise ValueError("request file must contain 24 unique, label-free raw requests")
    labels = {item["scenario_id"]: item for item in labels_in}
    if set(ids) != set(labels):
        raise ValueError("probe request and evaluator sidecar ids differ")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("diagnostic output directory must be empty")
    base = Path(config["base_snapshot"]).resolve(strict=True)
    bev, sequence_v2, sequence_v3, sequence_candidate, source_root = trainer._runtime(base, trainer._source_root())
    import torch
    import torch.nn.functional as F
    import peft
    if not torch.cuda.is_available():
        raise RuntimeError("the diagnostic requires the configured single CUDA device")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    # Production projection rebuilt directly from raw requests. Frozen sidecar
    # labels never enter the model state or tokenization path.
    # Some frozen labels carry a pair_id whose other member was not selected
    # by the fixed 24-row stratified probe. Keep the source/verdict only here;
    # these examples are CE-only singletons, with no fabricated pair objective.
    safe_labels = [{key: item[key] for key in ("scenario_id", "verdict", "source")
                    if key in item} for item in labels_in]
    raw_rows = [{"scenario_id": row["scenario_id"], "model_state": {}, "source": "default"}
                for row in request_rows]
    joined, target_map = normalize_corpus(raw_rows, request_rows, safe_labels)
    decider, optimizer, lora_config, trainable_names = trainer._load_model(
        bev, None, torch, peft, float(config["learning_rate"]), float(config["weight_decay"]))
    frozen_sha = trainer._module_tensor_sha256(decider.network.backbone)
    samples = []
    for row in joined:
        prepared = sequence_candidate.encode_candidate_input(row["request"], decider,
                                                              input_profile=PROFILE)
        state = prepared["model_state"]
        row["state"] = state
        encoded = prepared["encoded"]
        hashes = trainer._hash_example(encoded)
        if (hashes["encoded_sha256"] != prepared["encoded_input_sha256"]
                or hashes["input_ids_sha256"] != prepared["input_ids_sha256"]
                or hashes["attention_mask_sha256"] != prepared["attention_mask_sha256"]):
            raise RuntimeError("shared production encoder fingerprint mismatch")
        row.update({"keys": list(encoded["keys"]), "state_sha256": sha256_json(state),
                    "state_text_sha256": prepared["model_state_text_sha256"],
                    "question_sha256": prepared["question_sha256"],
                    "options_sha256": prepared["options_sha256"], **hashes,
                    "target": encoded["keys"].index(target_map[row["sample_id"]]["verdict"])})
        samples.append(row)
    assert_gemini_module_origins(source_root)
    source_indices = {source: [i for i, row in enumerate(samples) if row["source"] == source]
                      for source in ("fresh", "retained")}
    if any(len(indices) != 12 for indices in source_indices.values()):
        raise ValueError("the frozen diagnostic must contain 12 fresh and 12 retained TRAIN rows")
    training_units = [(i,) for i in range(24)]

    # Baseline is inference-only; no activation graphs are retained.
    baseline_losses, baseline_correct = [], 0
    with torch.inference_mode():
        for row in samples:
            logits = trainer._forward_one(decider, row, bev, torch)
            baseline_losses.append(float(F.cross_entropy(
                logits.unsqueeze(0), torch.tensor([row["target"]], device=logits.device)).cpu()))
            baseline_correct += int(int(logits.argmax().cpu()) == row["target"])
    baseline = {"role": "fresh_base_initialization_only", "count": 24,
                "correct": baseline_correct, "mean_cross_entropy": sum(baseline_losses) / 24,
                "assessment_path": "in-memory singleton; inference_mode; before optimizer update"}
    _write_json(output / "baseline.json", baseline)

    # A deterministic permutation changes order, not membership. Every update
    # backpropagates each of the same 24 rows immediately and steps once.
    eval_history, consecutive_success = [], 0
    temporary = output / "_assessor_candidate"
    temporary.mkdir()
    metadata = {**trainer._profile_metadata(PROFILE, sequence_v2, sequence_v3, config),
                "schema": "gemini_train_only_repeated_fit/1",
                "purpose": "fixed TRAIN-fit diagnostic only; never deployment or held-out evidence",
                "lora_config": lora_config, "trainable_parameter_names": trainable_names,
                "warm_checkpoint_sha256": None,
                "fresh_initialization_role": "pinned_base_pretrained_head_plus_new_lora",
                "training_input_manifest_sha256": config["probe_manifest_sha256"],
                "precision": "FP32 CUDA singleton training and evaluation; TF32 disabled",
                "microbatch_size": 1, "optimizer_gradient_accumulation_examples": 24,
                "promotion_eligible": False}
    _atomic_progress(output / "progress.json", {"status": "preflight_passed", "step": 0,
                                                 "baseline": baseline, "profile": PROFILE})
    for step in range(1, 257):
        # Shuffle singleton order; each update always covers 12 rows/source.
        unit_order = list(range(len(training_units)))
        random.Random(17 + step).shuffle(unit_order)
        update_units = [training_units[index] for index in unit_order]
        update = trainer._accumulated_optimizer_update(
            decider, samples, update_units, bev, torch, F, optimizer,
            contrast_weight=float(config["contrast_weight"]),
            contrast_margin=float(config["contrast_margin"]))
        if update["accumulated_examples"] != 24:
            raise RuntimeError("optimizer update did not consume exactly 24 singleton examples")
        source_counts = {source: sum(samples[index]["source"] == source
                                     for unit in update_units for index in unit)
                         for source in source_indices}
        if source_counts != {"fresh": 12, "retained": 12}:
            raise RuntimeError("optimizer update did not preserve the frozen 12/12 source mix")
        if step % 16 != 0 and step != 256:
            continue
        candidate = trainer._save_step(temporary, step=step, decider=decider, bev=bev,
                                       peft=peft, metadata={**metadata, "selected_step": step,
                                                            "optimizer_step": step,
                                                            "last_update_loss": update})
        assessment = _assessment(samples, target_map, candidate, bev, sequence_candidate,
                                 frozen_sha, decider.network)
        # Preserve parameters only in a deliberately non-loadable schema.
        payload = torch.load(candidate, map_location="cpu", weights_only=True)
        diagnostic_payload = {
            "schema": SNAPSHOT_SCHEMA, "diagnostic_only": True,
            "promotion_eligible": False, "checkpoint_sha256": assessment["checkpoint_sha256"],
            "step": step, "probe_manifest_sha256": config["probe_manifest_sha256"],
            "model_state_profile": PROFILE, "profile_metadata": metadata,
            "lora_config": lora_config, "lora_state_dict": payload["lora_state_dict"],
            "head_state_dict": payload["head_state_dict"],
        }
        snapshot = output / f"train-only-step-{step:04d}.pt"
        with snapshot.open("xb") as stream:
            torch.save(diagnostic_payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        entry = {"step": step, "snapshot": snapshot.name,
                 "snapshot_sha256": _digest_file(snapshot),
                 "update_loss": update, **{k: v for k, v in assessment.items() if k != "predictions"},
                 "predictions": assessment["predictions"]}
        eval_history.append(entry)
        _atomic_progress(output / "progress.json", {"status": "running", "step": step,
            "eval_history": eval_history, "consecutive_success_checks": consecutive_success})
        candidate.unlink()
        (temporary / candidate.with_suffix(".json").name).unlink(missing_ok=True)
        if memorization_gate(assessment["correct"], assessment["mean_cross_entropy"]):
            consecutive_success += 1
        else:
            consecutive_success = 0
        if consecutive_success >= 2:
            break
    temporary.rmdir()
    receipt = {"schema": "gemini_train_only_repeated_fit_receipt/1",
               "diagnostic_only": True, "promotion_eligible": False,
               "status": "memorized" if consecutive_success >= 2 else "cap_reached",
               "completed_optimizer_steps": step, "examples_per_update": 24,
               "examples_per_source_per_update": {"fresh": 12, "retained": 12},
               "contrastive_pairs_used": 0,
               "contrast_weight": float(config["contrast_weight"]),
               "evaluation_every_steps": 16, "success_rule": "24/24 correct and mean CE <= 0.05 at two consecutive checks",
               "consecutive_success_checks": consecutive_success, "baseline": baseline,
               "probe_manifest_sha256": config["probe_manifest_sha256"],
               "requests_sha256": config["requests_sha256"], "labels_sha256": config["labels_sha256"],
               "profile_metadata": metadata, "frozen_backbone_sha256": frozen_sha,
               "trainable_parameter_names": trainable_names, "history": eval_history}
    _write_json(output / "receipt.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run(args.config, request_path=args.requests, label_path=args.labels,
                 manifest_path=args.manifest, output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
