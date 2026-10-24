"""Three saved TRAIN prompts: teacher forcing versus actual production generation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--qwen-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve previous diagnostics")
    args.output.mkdir(parents=True)
    import torch
    from decision_factory.runtime_guard.authorization_actor import leading_json_command
    from decision_factory.runtime_guard.qwen import QwenProposer
    from decision_factory.training.actor_repair import encode_response
    from decision_factory.recovery.core import _strict_decision
    actor = QwenProposer(args.adapter, base_model_path=args.qwen_base, device="cuda",
                          proposal_mode="generate", max_new_tokens=64)
    runtime = actor.callback.runtime
    model, tokenizer = runtime.model, runtime.tokenizer
    rows = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()][:3]
    results = []
    for row in rows:
        expected = json.loads(row["target"])
        inputs, targets = encode_response(tokenizer, row["input"], row["target"])
        ids = torch.tensor([inputs], dtype=torch.long, device="cuda")
        prefix_length = len(inputs) - len(targets) + 1
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                           logits_to_keep=len(targets)).logits[0].float()
            predicted = logits.argmax(-1)
            labels = torch.tensor(targets, dtype=torch.long, device="cuda")
            loss = float(torch.nn.functional.cross_entropy(logits, labels).cpu())
            prefix = ids[:, :prefix_length]
            prefix_logits = model(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                                  use_cache=False, logits_to_keep=1).logits[0, -1].float()
            first_full = logits[0]
            prefix_top = int(prefix_logits.argmax())
            full_top = int(first_full.argmax())
        def generate():
            raw = actor.callback(row["input"])
            try:
                command, boundary = leading_json_command(raw)
                answer, error = _strict_decision(command)
            except ValueError as exc:
                answer, error, boundary = None, str(exc), None
            return {"raw": raw, "decision": answer, "error": error,
                    "decision_boundary": boundary, "correct": answer == expected}
        generation = generate()
        record = {"row_id": row["row_id"], "prefix_index": row.get("prefix_index"),
                  "trajectory_id": row.get("trajectory_id"), "expected": expected,
                  "prompt_sha256": hashlib.sha256(row["input"].encode()).hexdigest(),
                  "teacher_forced_ce": loss, "teacher_forced_correct_tokens": int((predicted == labels).sum()),
                  "target_tokens": len(targets), "teacher_forced_text": tokenizer.decode(predicted.cpu()),
                  "prefix_only_first_token": prefix_top, "teacher_forced_first_token": full_top,
                  "expected_first_token": targets[0],
                  "prefix_vs_teacher_forced_first_logit_maxdiff": float((prefix_logits - first_full).abs().max().cpu()),
                  "production_generation": generation}
        if not generation["correct"]:
            old = model.generation_config.use_cache
            try:
                model.generation_config.use_cache = False
                record["uncached_generation"] = generate()
            finally:
                model.generation_config.use_cache = old
        results.append(record)
        (args.output / "partial.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({k: record[k] for k in ("prefix_index", "teacher_forced_ce", "teacher_forced_correct_tokens", "target_tokens", "production_generation")} ), flush=True)
    (args.output / "complete.json").write_text(json.dumps({"schema": "actor_train_generation_probe/1",
        "scope": "three inspected TRAIN prompts, not held-out evidence", "actor": actor.provenance,
        "rows_sha256": hashlib.sha256(args.rows.read_bytes()).hexdigest(), "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
