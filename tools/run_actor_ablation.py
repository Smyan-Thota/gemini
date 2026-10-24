"""Bounded CPU-only actor diagnosis on existing public development episodes."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qwen-base", type=Path, required=True)
    parser.add_argument("--qwen-adapter", type=Path, default=ROOT / "adapters" / "h17")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--variants", nargs="+", choices=("base-finite", "h17-generate", "base-generate"),
                        default=["base-generate"])
    parser.add_argument("--prompt-profile", choices=("legacy", "compact"), default="legacy")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("actor diagnosis output already exists")
    args.output.mkdir(parents=True)
    import torch
    torch.set_num_threads(2)
    from decision_factory.runtime_guard.qwen import QwenProposer
    if args.prompt_profile == "compact":
        from decision_factory.runtime_guard.authorization_actor import AuthorizationProposer as QwenProposer
    from decision_factory.runtime_guard.multi_tool_host import run_multi_tool_episode
    actor = QwenProposer(args.qwen_adapter, device="cpu", base_model_path=args.qwen_base,
                          proposal_mode="finite_mean_logprob", max_new_tokens=64)
    cases = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()][:args.limit]
    configurations = {"base-finite": (False, "finite_mean_logprob"),
                      "h17-generate": (True, "generate"), "base-generate": (False, "generate")}
    variants = [(name, *configurations[name]) for name in args.variants]
    for name, enabled, mode in variants:
        actor.proposal_mode = mode
        with args.output.joinpath(name + ".jsonl").open("x", encoding="utf-8") as stream:
            for case in cases:
                context = contextlib.nullcontext() if enabled else actor.callback.runtime.model.disable_adapter()
                with context:
                    trace = run_multi_tool_episode(case, proposer=actor, assessor=None, factual_progress=True)
                trace["actor_variant"] = name
                trace["actor_provenance"] = {**actor.provenance, "adapter_active": enabled}
                trace["actual_model_outputs"] = [{"raw_output": row.get("raw_output"), "error": row.get("error"),
                                                   "decision_boundary": row.get("decision_boundary")}
                                                   for row in actor.calls[-len(trace["steps"]):]]
                stream.write(json.dumps(trace, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                print(name, case["case_id"], trace["terminal"], trace["committed_effects"], flush=True)
    receipt = {"schema": "actor_ablation/1", "scope": "two-case CPU development diagnostic",
               "variants": [v[0] for v in variants], "cases": [c["case_id"] for c in cases],
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "complete": True}
    args.output.joinpath("receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
