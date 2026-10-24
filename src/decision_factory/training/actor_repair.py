"""One bounded H17 adapter repair for tool grounding and factual progress.

Only supervised response tokens contribute loss. Training rows come from a
separate TRAIN-only corpus; this module does not load task evaluation keys.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import math
import time

QWEN_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
H17_SHA256 = "4c77c032bee13c08cd6077165414ddfc5ecbc5dacd8560a459ec4bd943f6e73a"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode_response(tokenizer, prompt: str, target: str, content_weighted: bool = False):
    chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prefix = tokenizer(chat, add_special_tokens=False)["input_ids"]
    response = tokenizer(target, add_special_tokens=False)["input_ids"]
    combined = tokenizer(chat + target, add_special_tokens=False)["input_ids"]
    if combined != prefix + response:
        raise ValueError("response retokenizes the production prompt boundary")
    if not response or tokenizer.eos_token_id is None:
        raise ValueError("response or EOS is unavailable")
    targets = response + [int(tokenizer.eos_token_id)]
    if len(prefix) + len(targets) > 2048:
        raise ValueError("training row exceeds supported context; no truncation allowed")
    if not content_weighted:
        return prefix + targets[:-1], targets
    mapping = tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
    if mapping["input_ids"] != response or len(mapping["offset_mapping"]) != len(response):
        raise ValueError("response offset mapping differs from exact target tokens")
    decoded = json.loads(target)
    spans = []
    for key, weight in (("action", 8.), ("type", 2.)):
        if key not in decoded:
            continue
        match = re.search(r'"' + key + r'"\s*:\s*"((?:\\.|[^"\\])*)"', target)
        if match is None or json.loads('"' + match.group(1) + '"') != decoded[key]:
            raise ValueError("critical decision span is not the exact JSON field value")
        spans.append((match.start(1), match.end(1), weight))
    weights, critical = [], []
    for start, end in mapping["offset_mapping"]:
        if not 0 <= start < end <= len(target):
            raise ValueError("invalid response token offset")
        overlaps = [weight for left, right, weight in spans if start < right and end > left]
        weights.append(max(overlaps, default=.25))
        critical.append(bool(overlaps))
    if not any(critical):
        raise ValueError("response has no critical decision token")
    return prefix + targets[:-1], targets, weights + [1.], critical + [False]


def response_loss(model, encoded, device="cuda", return_metrics=False):
    """Use only response-position logits, avoiding a full vocabulary/context tensor."""
    import torch
    inputs, targets = encoded[:2]
    ids = torch.tensor([inputs], dtype=torch.long, device=device)
    labels = torch.tensor(targets, dtype=torch.long, device=device)
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                   logits_to_keep=len(targets))
    logits = output.logits[0].float()
    if logits.shape[0] != len(targets):
        raise ValueError("model did not return exact completion positions")
    losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    weights = torch.tensor(encoded[2], dtype=torch.float32, device=device) if len(encoded) == 4 else torch.ones_like(losses)
    loss = (losses * weights).sum() / weights.sum()
    if not return_metrics:
        return loss
    critical = torch.tensor(encoded[3], dtype=torch.bool, device=device) if len(encoded) == 4 else torch.ones_like(labels, dtype=torch.bool)
    correct = logits.argmax(-1) == labels
    return loss, {"critical_correct_tokens": int(correct[critical].sum().detach().cpu()),
                  "critical_total_tokens": int(critical.sum().detach().cpu()),
                  "teacher_forced_critical_exact": bool(correct[critical].all().detach().cpu()),
                  "teacher_forced_sequence_exact": bool(correct.all().detach().cpu())}


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qwen-base", type=Path, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--initial-adapter-sha256", default=H17_SHA256)
    parser.add_argument("--content-weighted", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=0,
                        help="zero selects three complete epochs, capped at 256 updates")
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("preserve previous runs; choose a new output folder")
    if args.qwen_base.name != QWEN_REVISION or sha(args.initial_adapter / "adapter_model.safetensors") != args.initial_adapter_sha256.lower():
        raise ValueError("base or H17 initialization is not pinned")
    if not 0 <= args.updates <= 256 or not 1 <= args.accumulation <= 8:
        raise ValueError("repair exceeds the bounded protocol")
    adapter_config = json.loads((args.initial_adapter / "adapter_config.json").read_text())
    if adapter_config.get("r") != 8:
        raise ValueError("expected existing rank-eight H17 adapter")
    # A corpus manifest is mandatory; its file hash is bound into every run.
    manifest = json.loads(args.manifest.read_text())
    from decision_factory.runtime_guard import authorization_actor, multi_tool_host
    if (manifest.get("schema") != "authorization_actor_train_corpus/1"
            or manifest.get("files", {}).get("rows.jsonl", {}).get("sha256") != sha(args.rows)
            or manifest.get("renderer_sha256") != sha(authorization_actor.__file__)
            or manifest.get("host_sha256") != sha(multi_tool_host.__file__)):
        raise ValueError("TRAIN corpus or production prompt/host provenance changed")
    rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row.get("input"), str) or not isinstance(row.get("target"), str) for row in rows):
        raise ValueError("expected nonempty exact-prompt/completion training rows")
    if manifest.get("rows") != len(rows) or manifest["files"]["rows.jsonl"].get("count") != len(rows):
        raise ValueError("TRAIN corpus row count mismatch")
    args.updates = args.updates or min(256, math.ceil(3 * len(rows) / args.accumulation))
    from decision_factory.recovery.core import _strict_decision
    for row in rows:
        if _strict_decision(row["target"])[0] is None:
            raise ValueError("training target violates production decision protocol")
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    if not torch.cuda.is_available():
        raise RuntimeError("bounded actor training requires the available CUDA GPU")
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(str(args.qwen_base), local_files_only=True, trust_remote_code=False)
    encoded = [encode_response(tokenizer, row["input"], row["target"], args.content_weighted) for row in rows]
    base = Qwen3_5ForCausalLM.from_pretrained(str(args.qwen_base), dtype=torch.bfloat16,
                                             local_files_only=True, trust_remote_code=False)
    model = PeftModel.from_pretrained(base, str(args.initial_adapter), is_trainable=True, local_files_only=True)
    layers = sorted({int(match.group(1)) for name, p in model.named_parameters()
                     if p.requires_grad and (match := re.search(r"\.layers\.(\d+)\.", name))})
    selected_layers = layers[-2:]
    model.requires_grad_(False)
    for name, parameter in model.named_parameters():
        match = re.search(r"\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)\.lora_[AB]\.", name)
        if match and int(match.group(1)) in selected_layers:
            parameter.requires_grad_(True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # Non-reentrant checkpointing supports parameter gradients without forcing
    # the frozen embedding/early layers into the backward graph.
    if hasattr(model.get_base_model(), "_require_grads_hook"):
        model.disable_input_require_grads()
    model.to("cuda").train()
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise ValueError("only existing H17 LoRA parameters may be trained")
    args.output.mkdir(parents=True)
    config = {"schema": "qwen_actor_repair/1", "rows_sha256": sha(args.rows),
              "manifest_sha256": sha(args.manifest), "source_sha256": sha(__file__),
              "prompt_source_sha256": sha(authorization_actor.__file__),
              "prompt_profile": "compact_authorization_v1", "base_revision": QWEN_REVISION,
              "initial_adapter_sha256": args.initial_adapter_sha256.lower(), "row_count": len(rows),
              "loss_profile": "action8_type2_format0.25_eos1" if args.content_weighted else "uniform_completion_ce",
              "updates_cap": args.updates, "accumulation": args.accumulation,
              "learning_rate": args.learning_rate, "seed": args.seed,
              "trainable_parameters": sum(p.numel() for _, p in trainable),
              "trainable_layers": selected_layers, "trainable_names": [name for name, _ in trainable],
              "max_input_tokens": max(len(item[0]) for item in encoded),
              "precision": "bf16", "training_split": "TRAIN only; no task evaluation keys"}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    torch.cuda.reset_peak_memory_stats()
    smoke = response_loss(model, encoded[0])
    smoke.backward()
    gradients = [p.grad for _, p in trainable if p.grad is not None]
    if (not gradients or not all(torch.isfinite(g).all() for g in gradients)
            or not any(torch.count_nonzero(g) for g in gradients)):
        raise ValueError("actor smoke has missing, nonfinite or zero adapter gradients")
    (args.output / "smoke.json").write_text(json.dumps({"loss": float(smoke.detach().cpu()),
        "finite_gradient_tensors": len(gradients), "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "optimizer_updates": 0}, indent=2) + "\n")
    model.save_pretrained(args.output / "smoke-adapter")
    from safetensors.torch import load_file
    from peft import get_peft_model_state_dict
    loaded = load_file(args.output / "smoke-adapter" / "adapter_model.safetensors")
    state = get_peft_model_state_dict(model)
    if set(loaded) != set(state) or any(not torch.equal(loaded[key], state[key].detach().cpu()) for key in loaded):
        raise ValueError("saved adapter smoke does not reload exactly")
    model.zero_grad(set_to_none=True)
    del smoke, gradients
    if args.smoke_only:
        return 0
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=args.learning_rate, weight_decay=0.)
    order = list(range(len(encoded)))
    random.Random(args.seed).shuffle(order)
    started = time.perf_counter()
    examples, recent_losses = 0, []
    with (args.output / "progress.jsonl").open("x", encoding="utf-8") as log:
        for update in range(1, args.updates + 1):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(args.accumulation):
                item = encoded[order[examples % len(order)]]
                loss, fit = response_loss(model, item, return_metrics=True)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite actor training loss")
                losses.append(float(loss.detach().cpu()))
                (loss / args.accumulation).backward()
                examples += 1
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.)
            if not torch.isfinite(norm):
                raise ValueError("nonfinite actor adapter gradients")
            optimizer.step()
            mean_loss = sum(losses) / len(losses)
            recent_losses.append(mean_loss)
            record = {"updates": update, "examples": examples, "mean_completion_ce": mean_loss,
                      "last_example_fit": fit,
                      "wall_seconds": round(time.perf_counter() - started, 2)}
            log.write(json.dumps(record) + "\n"); log.flush(); os.fsync(log.fileno())
            print(json.dumps(record), flush=True)
            if update % 32 == 0:
                model.save_pretrained(args.output / f"checkpoint-{update:04d}")
            if (not args.content_weighted and update >= max(32, math.ceil(len(rows) / args.accumulation))
                    and sum(recent_losses[-8:]) / 8 < .03):
                break
    model.save_pretrained(args.output / "adapter")
    tokenizer.save_pretrained(args.output / "adapter")
    # Re-read the actual serialized adapter into the same pinned base and
    # compare an exact production training prompt before claiming portability.
    from peft import set_peft_model_state_dict
    model.eval()
    full_fit = []
    with torch.inference_mode():
        for row, item in zip(rows, encoded):
            fit_loss, metrics = response_loss(model, item, return_metrics=True)
            full_fit.append({"row_id": row["row_id"], "weighted_ce": float(fit_loss.cpu()), **metrics})
    (args.output / "train-teacher-forced-fit.json").write_text(json.dumps({
        "scope": "TRAIN teacher forcing, not greedy generation or held-out performance",
        "rows": len(full_fit), "critical_exact_rows": sum(r["teacher_forced_critical_exact"] for r in full_fit),
        "sequence_exact_rows": sum(r["teacher_forced_sequence_exact"] for r in full_fit),
        "critical_correct_tokens": sum(r["critical_correct_tokens"] for r in full_fit),
        "critical_total_tokens": sum(r["critical_total_tokens"] for r in full_fit), "details": full_fit}, indent=2) + "\n")
    smoke_ids, smoke_targets = encoded[0][:2]
    ids = torch.tensor([smoke_ids], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        before = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                       logits_to_keep=len(smoke_targets)).logits.detach().cpu().clone()
    serialized = load_file(args.output / "adapter" / "adapter_model.safetensors")
    if set(serialized) != set(get_peft_model_state_dict(model)):
        raise ValueError("final saved adapter parameter coverage mismatch")
    loaded_result = set_peft_model_state_dict(model, serialized)
    if loaded_result.unexpected_keys:
        raise ValueError("final adapter reload has unexpected parameters")
    with torch.inference_mode():
        after = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                      logits_to_keep=len(smoke_targets)).logits.detach().cpu()
    if not torch.equal(before, after):
        raise ValueError("saved/reloaded actor logits differ on exact prompt")
    (args.output / "receipt.json").write_text(json.dumps({**config, "status": "complete",
        "actual_updates": update, "examples": examples, "last_train_ce": mean_loss,
        "adapter_sha256": sha(args.output / "adapter" / "adapter_model.safetensors"),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "serialized_reload_logit_parity": True,
        "claim": "TRAIN optimization only; independent DEV generation remains required"}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
