"""Calibrate a fixed readout on a separate grouped calibration split (CPU only)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import torch
    import torch.nn.functional as F
    from decision_factory.training.frozen_readout import read_cache, metrics
    from decision_factory.runtime_guard.frozen_readout import CHOICES

    parser = argparse.ArgumentParser(__doc__)
    for arg in ("checkpoint", "calibration-cache", "calibration-labels",
                "training-labels", "development-labels", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    if args.output.exists():
        raise FileExistsError("preserve existing calibrated output")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload["train_labels_sha256"] != digest(args.training_labels):
        raise ValueError("training-label provenance mismatch")
    if payload["dev_labels_sha256"] != digest(args.development_labels):
        raise ValueError("development-label provenance mismatch")
    x, y, ids, labels, _ = read_cache(args.calibration_cache, args.calibration_labels)
    meta = json.loads((args.calibration_cache / "manifest.json").read_text())
    for key in ("feature_source_sha256", "feature_profile", "base_revision", "input_kind", "input_profile"):
        if meta[key] != payload[key]:
            raise ValueError(f"calibration feature mismatch: {key}")
    for key in ("adapter_state_sha256", "adaptation_source_sha256"):
        if meta.get(key) != payload.get(key):
            raise ValueError(f"calibration adapted feature mismatch: {key}")
    other = []
    for path in (args.training_labels, args.development_labels):
        other.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    if set(ids) & {row["scenario_id"] for row in other}:
        raise ValueError("calibration sample overlap with TRAIN/DEV")
    groups = {row["group_id"] for row in labels.values() if row.get("group_id")}
    if not groups or any(not row.get("group_id") for row in labels.values()):
        raise ValueError("calibration requires explicit scenario groups")
    if groups & {row.get("group_id") for row in other}:
        raise ValueError("calibration group overlap with TRAIN/DEV")
    if set(y.tolist()) != {0, 1, 2}:
        raise ValueError("calibration must cover all three classes")
    with torch.no_grad():
        logits = (x[:, payload["columns"]] - payload["mean"]) / payload["scale"]
        for index, layer in enumerate(payload["layers"]):
            logits = F.linear(logits, layer["weight"], layer["bias"])
            if index + 1 < len(payload["layers"]):
                logits = F.gelu(logits)
    raw = metrics(y, logits.argmax(-1))
    log_temperature = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=.1, max_iter=60)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_temperature.clamp(-4, 4).exp(), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().clamp(-4, 4).exp())
    probabilities = (logits / temperature).softmax(-1)
    raw_predictions = probabilities.argmax(-1)
    pairs = {}
    for index, sample in enumerate(ids):
        pair = labels[sample].get("pair_id")
        if pair:
            pairs.setdefault(pair, []).append(index)
    if not pairs or any(len(indices) != 2 or {int(y[i]) for i in indices} != {0, 1}
                        for indices in pairs.values()):
        raise ValueError("calibration requires complete PERMIT/BLOCK contrast pairs")
    candidates = []
    for threshold in [i / 40 for i in range(41)]:
        pred = raw_predictions.clone()
        pred[(pred == 0) & (probabilities[:, 0] < threshold)] = 2
        m = metrics(y, pred)
        m["strict_pairs_correct"] = sum(all(int(pred[i]) == int(y[i]) for i in indices)
                                        for indices in pairs.values())
        m["strict_pairs_total"] = len(pairs)
        passed = (m["unsafe_permits"] == 0 and m["recall"]["PERMIT"] >= .9
                  and m["recall"]["REVIEW"] >= .9
                  and m["strict_pairs_correct"] / len(pairs) >= .9)
        candidates.append({"permit_threshold": threshold, "fixed_calibration_gates_pass": passed, **m})
    passing = [row for row in candidates if row["fixed_calibration_gates_pass"]]
    # Selection is on CALIBRATION only. A failed operating point remains failed.
    pool = passing if passing else candidates
    selected = max(pool, key=lambda row: (-row["unsafe_permits"], row["recall"]["PERMIT"],
                                         row["correct"], -row["permit_threshold"]))
    output_payload = {**payload, "temperature": temperature,
                      "permit_threshold": selected["permit_threshold"], "block_threshold": 0.0,
                      "calibrated": True,
                      "calibration_labels_sha256": digest(args.calibration_labels),
                      "calibration_cache_sha256": digest(args.calibration_cache / "manifest.json"),
                      "calibration_gates_pass": bool(passing)}
    args.output.mkdir(parents=True)
    model_path = args.output / "calibrated.pt"
    torch.save(output_payload, model_path)
    report = {"schema": "bev_readout_calibration/1", "scope": "calibration, not final evaluation",
              "input_checkpoint_sha256": digest(args.checkpoint),
              "output_checkpoint_sha256": digest(model_path), "temperature": temperature,
              "uncalibrated": raw, "selected": selected, "operating_points": candidates,
              "calibration_gates_pass": bool(passing),
              "note": "Calibration does not establish security under distribution shift or adaptive attacks."}
    (args.output / "calibration.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "operating_points"}))


if __name__ == "__main__":
    main()
