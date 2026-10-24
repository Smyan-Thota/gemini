"""Run local Qwen plus a frozen Bev readout in the synthetic multi-tool host."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

from .multi_tool_host import digest, run_multi_tool_episode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--qwen-base", type=Path, required=True)
    parser.add_argument("--qwen-adapter", type=Path)
    parser.add_argument("--qwen-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--qwen-proposal-mode", choices=("finite_mean_logprob", "generate"),
                        default="finite_mean_logprob")
    parser.add_argument("--disable-qwen-adapter", action="store_true",
                        help="diagnostic base-Qwen comparison using the same loaded base weights")
    parser.add_argument("--actor-prompt-profile", choices=("legacy", "compact", "native"), default="legacy")
    parser.add_argument("--readout-checkpoint", type=Path)
    parser.add_argument("--readout-sha256")
    parser.add_argument("--readout-kind", choices=("frozen", "adapted"), default="frozen")
    parser.add_argument("--mode", choices=("qwen-bev", "qwen-alone", "matched"), default="qwen-bev",
                        help="matched runs the same actor, menus and host in both conditions")
    parser.add_argument("--bev-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--factual-progress", action="store_true",
                        help="render host-authored committed totals without computing a desired action")
    args = parser.parse_args(argv)
    native = args.actor_prompt_profile == "native"
    if native and args.disable_qwen_adapter:
        parser.error("the native profile already uses base Qwen without an adapter")
    if not native and args.qwen_adapter is None:
        parser.error("legacy and compact profiles require --qwen-adapter")
    if args.out.exists() or args.out.with_suffix(args.out.suffix + ".receipt.json").exists():
        raise FileExistsError("choose a new trace output; existing results are preserved")
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("input contains no public episodes")
    from .qwen import QwenProposer
    if args.actor_prompt_profile == "compact":
        from .authorization_actor import AuthorizationProposer as QwenProposer
    readout = None
    if args.mode != "qwen-alone":
        if args.readout_checkpoint is None or args.readout_sha256 is None:
            raise ValueError("guarded/matched mode requires a readout checkpoint and SHA256")
        if args.readout_kind == "adapted":
            from .adapted_readout import load_adapted_readout as loader
        else:
            from .frozen_readout import load_readout as loader
        readout = loader(args.readout_checkpoint, expected_sha256=args.readout_sha256, device=args.bev_device)
    if native:
        from .native_actor import NativeQwenProposer
        proposer = NativeQwenProposer(args.qwen_base, device=args.qwen_device)
    else:
        proposer = QwenProposer(args.qwen_adapter, device=args.qwen_device,
                                base_model_path=args.qwen_base, proposal_mode=args.qwen_proposal_mode)
    adapter_active = not native and not args.disable_qwen_adapter
    started = time.perf_counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    traces = []
    # Persist each completed episode so a later interruption cannot discard it.
    with args.out.open("x", encoding="utf-8") as stream:
        modes = ("qwen-alone", "qwen-bev") if args.mode == "matched" else (args.mode,)
        for index, case in enumerate(rows, 1):
            for mode in modes:
                context = (proposer.callback.runtime.model.disable_adapter()
                           if args.disable_qwen_adapter else contextlib.nullcontext())
                with context:
                    trace = run_multi_tool_episode(case, proposer=proposer,
                                                    assessor=readout.assess_scene if mode == "qwen-bev" else None,
                                                    max_steps=args.max_steps, factual_progress=args.factual_progress)
                trace["actor_proposal_mode"] = "native_function_call" if native else args.qwen_proposal_mode
                trace["qwen_adapter_active"] = adapter_active
                traces.append(trace)
                stream.write(json.dumps(trace, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                print(f"{index}/{len(rows)} {mode} {trace['case_id']} {trace['terminal']} effects={trace['executed_operations']}", flush=True)
    receipt = {"schema": "multi_tool_run_receipt/1", "status": "complete",
               "episodes": len(traces), "conditions": args.mode,
               "factual_progress": args.factual_progress, "input_sha256": digest(rows),
               "trace_sha256": digest(traces),
               "qwen": {**proposer.provenance, "adapter_active": adapter_active},
               "actor_prompt_profile": args.actor_prompt_profile,
               "readout_sha256": args.readout_sha256, "bev_device": args.bev_device,
               "readout_kind": args.readout_kind,
               "bev_base_revision": readout.payload["base_revision"] if readout is not None else None,
               "feature_profile": readout.payload["feature_profile"] if readout is not None else None,
               "source_files": {name: {"path": str(Path(__file__).parent / name),
                                        "sha256": hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()}
                                for name in ("multi_tool_cli.py", "multi_tool_host.py", "qwen.py", "frozen_readout.py",
                                             "native_actor.py", "authorization_actor.py", "adapted_readout.py")},
               "python": sys.version, "platform": platform.platform(),
               "wall_seconds": round(time.perf_counter() - started, 3),
               "external_effects": "none; synthetic ledger only"}
    args.out.with_suffix(args.out.suffix + ".receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
