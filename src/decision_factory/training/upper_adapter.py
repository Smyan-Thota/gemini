"""Bounded supervised upper-layer Bev adaptation on verified TRAIN variants."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

from .frozen_readout import sha, write_json, metrics, pair_metrics
from .protocol import load_jsonl
from decision_factory.runtime_guard import bev, frozen_readout, adapted_readout


def main():
    import peft
    import torch
    import torch.nn.functional as F
    parser = argparse.ArgumentParser(__doc__)
    for name in ["train-states", "train-labels", "augmented-states", "augmented-labels",
                 "dev-states", "dev-labels", "initial-readout", "output"]:
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    random.seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    source = torch.load(args.initial_readout, map_location="cpu", weights_only=True)
    expected_source = {"schema": frozen_readout.SCHEMA,
                       "feature_source_sha256": sha(frozen_readout.__file__),
                       "feature_profile": frozen_readout.FEATURE_PROFILE,
                       "base_revision": bev.MODEL_REVISION, "base_weight_sha256": bev.MODEL_SHA256,
                       "classes": list(frozen_readout.CHOICES), "input_kind": "states",
                       "input_profile": "multi_tool_scene_v2"}
    if any(source.get(k) != v for k, v in expected_source.items()):
        raise ValueError("initial readout provenance differs")
    rows = load_jsonl(args.train_states) + load_jsonl(args.augmented_states)
    labels = {row["scenario_id"]: row for path in [args.train_labels, args.augmented_labels] for row in load_jsonl(path)}
    dev = load_jsonl(args.dev_states)
    dev_labels = {row["scenario_id"]: row for row in load_jsonl(args.dev_labels)}
    if len(rows) != 1152 or len(labels) != 1152 or len(dev) != 144:
        raise ValueError("frozen protocol requires576original+576verifiedvariants and144DEV")
    if {row["scenario_id"] for row in rows} != set(labels) or {row["scenario_id"] for row in dev} != set(dev_labels):
        raise ValueError("model scenes and label sidecars differ")
    if {row["group_id"] for row in labels.values()} & {row["group_id"] for row in dev_labels.values()}:
        raise ValueError("TRAIN/DEV group overlap")
    for row in rows + dev:
        if set(row) - {"scenario_id", "source", "model_state"}:
            raise ValueError("model state row has unexpected metadata")
    decider = bev._load_real_decider()
    bev._cached_decider = None
    decider.network.to("cuda").eval().requires_grad_(False)
    decider.device = torch.device("cuda")
    # A real baseline and gradient smoke precede optimizer updates.
    first = frozen_readout.encode_scene(decider, rows[0]["model_state"])
    baseline, _ = frozen_readout.extract_features(decider, first, "cuda")
    adapter_config = adapted_readout.add_upper_adapter(decider)
    layers = []
    for index, layer in enumerate(source["layers"]):
        linear = torch.nn.Linear(layer["weight"].shape[1], layer["weight"].shape[0])
        linear.load_state_dict(layer)
        layers.append(linear)
        if index < len(source["layers"]) - 1:
            layers.append(torch.nn.GELU())
    head = torch.nn.Sequential(*layers).to("cuda").eval()
    mean, scale = source["mean"].cuda(), source["scale"].cuda()
    columns = source["columns"]
    def forward(encoded, training=False):
        feature = adapted_readout.feature_forward(decider, encoded, "cuda", training=training)
        return head((feature[columns] - mean) / scale)
    actual = adapted_readout.feature_forward(decider, first, "cuda", training=False).detach().cpu()
    if not torch.allclose(baseline, actual, atol=1e-5, rtol=1e-5):
        raise ValueError("zero-adapter feature parity failed")
    adapter_parameters = [p for name, p in decider.network.backbone.named_parameters() if p.requires_grad]
    if not adapter_parameters:
        raise ValueError("no trainable adapters")
    if any(p.requires_grad and "lora_" not in name for name, p in decider.network.backbone.named_parameters()):
        raise ValueError("unexpected trainable backbone parameter")
    label = torch.tensor([frozen_readout.CHOICES.index(labels[rows[0]["scenario_id"]]["verdict"])], device="cuda")
    F.cross_entropy(forward(first, True).unsqueeze(0), label).backward()
    finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in adapter_parameters)
    nonzero = sum(p.grad is not None and bool((p.grad != 0).any()) for p in adapter_parameters)
    if not finite or not nonzero:
        raise ValueError("adapter gradient smoke failed")
    for p in adapter_parameters + list(head.parameters()):
        p.grad = None
    del baseline, actual
    write_json(output / "smoke.json", {"feature_parity": True, "finite_gradients": finite,
               "nonzero_adapter_gradients": nonzero, "adapter_config": adapter_config,
               "trainable_adapter_parameters": sum(p.numel() for p in adapter_parameters),
               "readout_parameters": sum(p.numel() for p in head.parameters())})
    config = {"schema": "bev_upper_adapter_protocol/1", "seed": 17,
              "initial_cap_updates": 144, "maximum_cap_updates": 288,
              "examples_per_update": 24, "adapter_lr": .0001, "head_lr": .001,
              "precision": "fp32", "input_batch_size": 1,
              "extension": "at144only:trainCEimproves5percentandDEVaccuracyincreasesorunsafeimproveswithoutP/Rloss",
              "inputs": {name: {"path": str(getattr(args, name)), "sha256": sha(getattr(args, name))}
                         for name in ["train_states", "train_labels", "augmented_states", "augmented_labels",
                                      "dev_states", "dev_labels", "initial_readout"]},
              "trainer_source_sha256": sha(__file__), "adaptation_source_sha256": sha(adapted_readout.__file__),
              "feature_source_sha256": sha(frozen_readout.__file__)}
    write_json(output / "protocol.json", config)
    encoded_train = [(row["scenario_id"], frozen_readout.encode_scene(decider, row["model_state"])) for row in rows]
    encoded_dev = [(row["scenario_id"], frozen_readout.encode_scene(decider, row["model_state"])) for row in dev]
    optimizer = torch.optim.AdamW([{"params": adapter_parameters, "lr": .0001},
                                  {"params": head.parameters(), "lr": .001}], weight_decay=.001)
    history = []
    started = time.perf_counter()
    for epoch in range(1, 7):
        indices = list(range(len(rows)))
        random.Random(17 + epoch).shuffle(indices)
        loss_sum = 0.0
        for offset in range(0, len(indices), 24):
            optimizer.zero_grad(set_to_none=True)
            for index in indices[offset:offset + 24]:
                sample, encoded = encoded_train[index]
                target = torch.tensor([frozen_readout.CHOICES.index(labels[sample]["verdict"])], device="cuda")
                loss = F.cross_entropy(forward(encoded, True).unsqueeze(0), target)
                loss_sum += float(loss.detach())
                (loss / 24).backward()
            torch.nn.utils.clip_grad_norm_(adapter_parameters + list(head.parameters()), 1.0)
            optimizer.step()
            update = (epoch - 1) * 48 + offset // 24 + 1
            if update % 8 == 0:
                print(json.dumps({"updates": update, "epoch": epoch, "examples": update * 24,
                                  "mean_train_ce": loss_sum / (offset + 24),
                                  "seconds": time.perf_counter() - started}), flush=True)
        predictions, logits = [], []
        with torch.inference_mode():
            for sample, encoded in encoded_dev:
                score = forward(encoded)
                predictions.append(int(score.argmax()))
                logits.append(score.detach().cpu().tolist())
        ids = [row[0] for row in encoded_dev]
        y = torch.tensor([frozen_readout.CHOICES.index(dev_labels[s]["verdict"]) for s in ids])
        pred = torch.tensor(predictions)
        m = metrics(y, pred)
        m.update(pair_metrics(ids, dev_labels, pred))
        m.update({"epoch": epoch, "updates": epoch * 48, "examples": epoch * 1152,
                  "train_ce": loss_sum / 1152})
        payload = {**source, "schema": adapted_readout.SCHEMA,
                   "adaptation_source_sha256": sha(adapted_readout.__file__),
                   "adapter_config": adapter_config,
                   "adapter_state": {k: v.detach().cpu().clone() for k, v in peft.get_peft_model_state_dict(decider.network.backbone).items()},
                   "layers": [{"weight": layer.weight.detach().cpu().clone(),
                               "bias": layer.bias.detach().cpu().clone()} for layer in head if isinstance(layer, torch.nn.Linear)],
                   "temperature": 1.0, "permit_threshold": 0.0, "block_threshold": 0.0,
                   "calibrated": False, "selection": m, "adaptation_protocol_sha256": sha(output / "protocol.json")}
        path = output / f"step-{epoch * 48:04d}.pt"
        payload["adapter_state_sha256"] = adapted_readout.adapter_digest(payload["adapter_state"])
        torch.save(payload, path)
        m["checkpoint_sha256"] = sha(path)
        write_json(output / f"step-{epoch * 48:04d}.predictions.json", [
            {"scenario_id": s, "expected": frozen_readout.CHOICES[int(y[i])],
             "predicted": frozen_readout.CHOICES[predictions[i]], "logits": logits[i]} for i, s in enumerate(ids)])
        history.append(m)
        write_json(output / "metrics.json", history)
        print(json.dumps(m), flush=True)
        gate = (m["unsafe_permits"] == 0 and m["recall"]["PERMIT"] >= .9
                and m["recall"]["REVIEW"] >= .9 and m["strict_pairs_correct"] / m["strict_pairs_total"] >= .9)
        if gate:
            break
        if epoch == 3:
            prev = history[-2]
            improving_loss = m["train_ce"] <= .95 * prev["train_ce"]
            improving_dev = m["correct"] > prev["correct"] or (
                m["unsafe_permits"] < prev["unsafe_permits"] and
                m["recall"]["PERMIT"] >= prev["recall"]["PERMIT"] and
                m["recall"]["REVIEW"] >= prev["recall"]["REVIEW"])
            write_json(output / "extension-decision.json", {"extend": improving_loss and improving_dev,
                       "improving_train_ce": improving_loss, "improving_dev": improving_dev})
            if not improving_loss or not improving_dev:
                break
    write_json(output / "complete.json", {"updates": history[-1]["updates"],
               "examples": history[-1]["examples"], "development_gates_pass": gate,
               "history": history, "seconds": time.perf_counter() - started})


if __name__ == "__main__":
    main()
