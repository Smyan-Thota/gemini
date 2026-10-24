"""Cache singleton frozen Bev features, then fit bounded CPU risk readouts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from .protocol import load_jsonl
from decision_factory.runtime_guard import bev, frozen_readout as runtime


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def cache(args):
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    provenance = {"schema": "bev_feature_cache/1", "input_sha256": sha(args.input),
                  "input_kind": args.kind, "input_profile": args.profile,
                  "feature_profile": runtime.FEATURE_PROFILE,
                  "feature_source_sha256": sha(runtime.__file__),
                  "base_revision": bev.MODEL_REVISION, "base_weight_sha256": bev.MODEL_SHA256,
                  "dtype": "float32", "batch_size": 1}
    metadata = out / "manifest.json"
    if metadata.exists() and json.loads(metadata.read_text()) != provenance:
        raise ValueError("existing cache provenance differs")
    write_json(metadata, provenance)
    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[:args.limit]
    decider = bev._load_real_decider()
    decider.network.eval().requires_grad_(False).to(args.device)
    decider.device = torch.device(args.device)
    started = time.perf_counter()
    for start in range(0, len(rows), 32):
        path = out / f"chunk-{start:06d}.pt"
        if path.exists():
            continue
        features, logits, ids, token_hashes = [], [], [], []
        for row in rows[start:start + 32]:
            if args.kind == "requests":
                from decision_factory.runtime_guard.sequence_candidate import encode_candidate_input
                encoded = encode_candidate_input(row["guard_request"], decider,
                                                 input_profile=args.profile)["encoded"]
            else:
                encoded = runtime.encode_scene(decider, row["model_state"])
            x, scores = runtime.extract_features(decider, encoded, args.device)
            features.append(x)
            logits.append(scores)
            ids.append(row["scenario_id"])
            token_hashes.append(hashlib.sha256(json.dumps(encoded["input_ids"]).encode()).hexdigest())
        temporary = path.with_suffix(".tmp")
        torch.save({"ids": ids, "features": torch.stack(features), "logits": torch.stack(logits),
                    "input_token_hashes": token_hashes}, temporary)
        os.replace(temporary, path)
        print(json.dumps({"cached": min(start + 32, len(rows)), "total": len(rows),
                          "seconds": round(time.perf_counter() - started, 1)}), flush=True)
    write_json(out / "complete.json", {"rows": len(rows), "chunks": (len(rows) + 31) // 32,
                                       "seconds": time.perf_counter() - started})


def read_cache(directory, labels_path):
    import torch
    root = Path(directory)
    complete = json.loads((root / "complete.json").read_text())
    chunks = [torch.load(path, weights_only=True) for path in sorted(root.glob("chunk-*.pt"))]
    ids = [sample for chunk in chunks for sample in chunk["ids"]]
    labels = {row["scenario_id"]: row for row in load_jsonl(labels_path)}
    if len(ids) != complete["rows"] or len(set(ids)) != len(ids) or set(ids) != set(labels):
        raise ValueError("cache and labels must match exactly")
    x = torch.cat([chunk["features"] for chunk in chunks])
    y = torch.tensor([runtime.CHOICES.index(labels[sample]["verdict"]) for sample in ids])
    logits = torch.cat([chunk["logits"] for chunk in chunks])
    return x, y, ids, labels, logits


def metrics(y, pred):
    import torch
    return {"correct": int((y == pred).sum()), "total": len(y),
            "unsafe_permits": int(((y != 0) & (pred == 0)).sum()),
            "confusion": [[int(((y == a) & (pred == b)).sum()) for b in range(3)] for a in range(3)],
            "recall": {key: float((pred[y == i] == i).float().mean())
                       if int((y == i).sum()) else None for i, key in enumerate(runtime.CHOICES)},
            "reviews": int((pred == 2).sum())}


def pair_metrics(ids, labels, predicted):
    groups = {}
    for index, sample in enumerate(ids):
        pair_id = labels[sample].get("pair_id")
        if pair_id:
            groups.setdefault(pair_id, []).append(index)
    pairs = [indices for indices in groups.values()
             if len(indices) == 2 and {labels[ids[i]]["verdict"] for i in indices} == {"PERMIT", "BLOCK"}]
    correct = sum(all(runtime.CHOICES[int(predicted[i])] == labels[ids[i]]["verdict"] for i in indices)
                  for indices in pairs)
    return {"strict_pairs_correct": correct, "strict_pairs_total": len(pairs)}


def fit(args):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(8)
    torch.manual_seed(17)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise ValueError("results already exist; preserve prior runs")
    train, target, train_ids, train_labels, _ = read_cache(args.train_cache, args.train_labels)
    dev, dev_target, dev_ids, dev_labels, baseline = read_cache(args.dev_cache, args.dev_labels)
    if set(train_ids) & set(dev_ids):
        raise ValueError("TRAIN/DEV ID overlap")
    groups_train = {row["group_id"] for row in train_labels.values() if row.get("group_id")}
    groups_dev = {row["group_id"] for row in dev_labels.values() if row.get("group_id")}
    if groups_train & groups_dev:
        raise ValueError("TRAIN/DEV group overlap")
    meta = json.loads((Path(args.train_cache) / "manifest.json").read_text())
    devmeta = json.loads((Path(args.dev_cache) / "manifest.json").read_text())
    if any(meta[k] != devmeta[k] for k in ("feature_source_sha256", "feature_profile", "base_revision", "input_kind", "input_profile")):
        raise ValueError("feature profiles differ")
    results = {"schema": "bev_readout_experiment/1", "evaluation_scope": "DEV, not independent final",
               "training_rows": len(target), "dev_rows": len(dev_target),
               "original_bev": metrics(dev_target, baseline.argmax(-1)), "candidates": []}
    best_key = None
    best_accuracy = {}
    for mode, hidden, decay in [("answer", 0, .01), ("all", 0, .01), ("all", 64, .001)]:
        columns = runtime.feature_columns(train.shape[1], mode)
        mean = train[:, columns].mean(0)
        scale = train[:, columns].std(0).clamp_min(.05)
        x = (train[:, columns] - mean) / scale
        z = (dev[:, columns] - mean) / scale
        torch.manual_seed(17)
        model = (torch.nn.Sequential(torch.nn.Linear(x.shape[1], hidden), torch.nn.GELU(),
                                      torch.nn.Linear(hidden, 3)) if hidden else
                 torch.nn.Sequential(torch.nn.Linear(x.shape[1], 3)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003 if hidden else .01, weight_decay=decay)
        best_loss = float("inf")
        stagnant = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            for indices in torch.randperm(len(x)).split(256):
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x[indices]), target[indices])
                loss.backward()
                optimizer.step()
            if epoch % 10 and epoch != args.epochs:
                continue
            model.eval()
            with torch.inference_mode():
                train_loss = float(F.cross_entropy(model(x), target))
                dev_scores = model(z)
                pred = dev_scores.argmax(-1)
                m = metrics(dev_target, pred)
                m.update(pair_metrics(dev_ids, dev_labels, pred))
            log = {"mode": mode, "hidden": hidden, "epoch": epoch,
                   "train_ce": train_loss, "train_correct": int((model(x).argmax(-1) == target).sum()),
                   **m}
            print(json.dumps(log), flush=True)
            # Checkpoint choice is explicitly DEV-selected; final is not used.
            gate = (m["unsafe_permits"] == 0 and m["recall"]["PERMIT"] >= .9
                    and m["recall"]["REVIEW"] >= .9 and m["strict_pairs_total"] > 0
                    and m["strict_pairs_correct"] / m["strict_pairs_total"] >= .9)
            key = (int(gate), m["correct"] - 2 * m["unsafe_permits"], m["correct"], -m["unsafe_permits"])
            architecture = f"{mode}-{hidden}"
            accuracy_key = (m["correct"], -m["unsafe_permits"])
            safety_improved = best_key is None or key > best_key
            accuracy_improved = architecture not in best_accuracy or accuracy_key > best_accuracy[architecture]
            if safety_improved or accuracy_improved:
                layers = [{"weight": layer.weight.detach().clone(), "bias": layer.bias.detach().clone()}
                          for layer in model if isinstance(layer, torch.nn.Linear)]
                payload = {"schema": runtime.SCHEMA, "feature_profile": runtime.FEATURE_PROFILE,
                           "feature_source_sha256": sha(runtime.__file__),
                           "base_revision": bev.MODEL_REVISION, "base_weight_sha256": bev.MODEL_SHA256,
                           "classes": list(runtime.CHOICES), "columns": columns,
                           "mean": mean, "scale": scale, "layers": layers,
                           "temperature": 1.0, "permit_threshold": 0.0, "block_threshold": 0.0,
                           "input_kind": meta["input_kind"], "input_profile": meta["input_profile"],
                           "train_cache_sha256": sha(Path(args.train_cache) / "manifest.json"),
                           "train_labels_sha256": sha(args.train_labels),
                           "dev_labels_sha256": sha(args.dev_labels),
                           "selection": log, "calibrated": False}
                predictions = [
                    {"scenario_id": sample, "expected": runtime.CHOICES[int(dev_target[i])],
                     "predicted": runtime.CHOICES[int(pred[i])], "logits": dev_scores[i].tolist()}
                    for i, sample in enumerate(dev_ids)]
                if safety_improved:
                    best_key = key
                    torch.save(payload, out / "selected.pt")
                    write_json(out / "selected-predictions.json", predictions)
                    write_json(out / "selection.json", {**log, "checkpoint_sha256": sha(out / "selected.pt")})
                if accuracy_improved:
                    best_accuracy[architecture] = accuracy_key
                    torch.save(payload, out / f"accuracy-{architecture}.pt")
                    write_json(out / f"accuracy-{architecture}.predictions.json", predictions)
                    write_json(out / f"accuracy-{architecture}.json", {**log, "checkpoint_sha256": sha(out / f"accuracy-{architecture}.pt")})
            if train_loss < best_loss - 1e-5:
                best_loss, stagnant = train_loss, 0
            else:
                stagnant += 1
            if stagnant >= 5 or train_loss < .005:
                break
        results["candidates"].append(log)
    results["selected"] = json.loads((out / "selection.json").read_text())
    write_json(out / "results.json", results)


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--input", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--kind", choices=["requests", "states"], default="states")
    c.add_argument("--profile", default="operation_facts_v3")
    c.add_argument("--device", default="cuda")
    c.add_argument("--limit", type=int)
    f = sub.add_parser("fit")
    for name in ["train-cache", "train-labels", "dev-cache", "dev-labels", "output"]:
        f.add_argument("--" + name, required=True)
    f.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    (cache if args.command == "cache" else fit)(args)


if __name__ == "__main__":
    main()
