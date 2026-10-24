"""Run a bounded multi-step Qwen -> Bev -> in-memory host episode."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ADAPTER = ROOT / "adapters" / "h17"
DEFAULT_SEQUENCE_CHECKPOINT = ROOT / "checkpoints" / "sequence-bev-v7-experimental.pt"
DEFAULT_LEGACY_CHECKPOINT = ROOT / "checkpoints" / "legacy-v4-bev.pt"
DEFAULT_INPUT = ROOT / "examples" / "sequence-host-demo.jsonl"
QWEN_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_cases(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        allowed = {"schema", "case_id", "task", "host_policy", "initial_observations",
                   "prior_actions", "virtual_files", "action_menu", "turn_action_menus",
                   "max_steps", "environment_schedule"}
        if not isinstance(row, dict) or row.get("schema") != "sequence_host_demo_case/1":
            raise ValueError(f"input row {number} must use sequence_host_demo_case/1")
        if set(row) - allowed or not isinstance(row.get("case_id"), str):
            raise ValueError(f"input row {number} contains unsupported fields")
        if (not isinstance(row.get("task"), dict) or not isinstance(row.get("host_policy"), dict)
                or not isinstance(row.get("initial_observations", []), list)
                or not isinstance(row.get("prior_actions", []), list)
                or not isinstance(row.get("virtual_files"), dict)):
            raise ValueError(f"input row {number} has invalid task, policy, history, or file state")
        if "max_steps" in row and (type(row["max_steps"]) is not int or not 1 <= row["max_steps"] <= 32):
            raise ValueError(f"input row {number} max_steps must be between 1 and 32")
        if not isinstance(row.get("action_menu"), list) and not isinstance(row.get("turn_action_menus"), list):
            raise ValueError(f"input row {number} needs action_menu or turn_action_menus")
        from .sequence_host import validate_environment_schedule
        validate_environment_schedule(row.get("environment_schedule"))
        rows.append(row)
    if not rows:
        raise ValueError("input contains no episodes")
    return rows


def planned_steps(case: dict[str, Any], requested: int | None) -> int:
    policy = case["host_policy"]
    limit = requested or case.get("max_steps") or policy.get("max_steps") or 4
    if type(limit) is not int or not 1 <= limit <= 32:
        raise ValueError("episode step limit must be between 1 and 32")
    menus = case.get("turn_action_menus")
    return min(limit, len(menus)) if isinstance(menus, list) else limit


def run_case(case: dict[str, Any], *, proposer: Any, sequence_checkpoint: Path,
             legacy_checkpoint: Path, max_steps: int | None = None) -> dict[str, Any]:
    from .host import GuardHost
    from .sequence import evaluate_routed_proposal
    from .sequence_host import run_sequence_episode

    base_request = {
        "schema": "bev_guard_request/1",
        "user_request": case["task"],
        "host_policy": case["host_policy"],
        "prior_actions": case.get("prior_actions", []),
        "observations": case.get("initial_observations", []),
        "proposal": {},
    }

    def assess(request: dict[str, Any], *, device: str = "cuda",
               head_checkpoint: str | Path | None = None) -> dict[str, Any]:
        return evaluate_routed_proposal(
            request,
            sequence_checkpoint=head_checkpoint or sequence_checkpoint,
            legacy_checkpoint=legacy_checkpoint,
            sequence_device=device,
        )

    legacy_route = "workflow_rules" not in case["host_policy"]
    host = GuardHost(request=base_request, initial_files=case["virtual_files"],
                     mode="qwen-bev", bev_evaluator=assess,
                     bev_head_checkpoint=str(sequence_checkpoint),
                     bev_device="cpu" if legacy_route else "cuda",
                     profile="legacy_v4_cpu" if legacy_route else "sequence")
    trace = run_sequence_episode(case, proposer=proposer, host=host, max_steps=max_steps)
    trace["qwen_provenance"] = proposer.provenance
    return trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, required=True,
                        help="new JSONL trace path; a sibling .receipt.json is written")
    parser.add_argument("--qwen-base", type=Path, required=True,
                        help="existing local Qwen snapshot at the adapter-pinned revision")
    parser.add_argument("--qwen-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--bev-device", choices=("cuda",), default="cuda",
                        help="sequence Bev is supported only on CUDA FP32")
    parser.add_argument("--sequence-checkpoint", type=Path, default=DEFAULT_SEQUENCE_CHECKPOINT)
    parser.add_argument("--legacy-checkpoint", type=Path, default=DEFAULT_LEGACY_CHECKPOINT)
    parser.add_argument("--max-steps", type=int,
                        help="maximum Qwen decisions per episode; default is bounded by host policy")
    args = parser.parse_args(argv)
    source = args.input.expanduser().resolve(strict=True)
    output = args.out.expanduser().resolve()
    qwen_base = args.qwen_base.expanduser().resolve(strict=True)
    sequence_checkpoint = args.sequence_checkpoint.expanduser().resolve(strict=True)
    legacy_checkpoint = args.legacy_checkpoint.expanduser().resolve(strict=True)
    if output.exists() or output.with_suffix(output.suffix + ".receipt.json").exists():
        raise FileExistsError("trace or receipt already exists; choose a new output path")
    if qwen_base.name != QWEN_REVISION:
        raise ValueError("Qwen snapshot directory must match the pinned revision")
    if args.max_steps is not None and not 1 <= args.max_steps <= 32:
        raise ValueError("--max-steps must be between 1 and 32")
    if args.bev_device != "cuda":
        raise ValueError("the sequence Bev profile requires CUDA FP32")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for sequence Bev but is unavailable")
    cases = read_cases(source)
    sys.path.insert(0, str(ROOT / "src"))
    from decision_factory.runtime_guard import bev
    from decision_factory.runtime_guard.qwen import QwenProposer
    from decision_factory.runtime_guard.sequence import SEQUENCE_CHECKPOINT_SHA256

    if sha256_file(sequence_checkpoint) != SEQUENCE_CHECKPOINT_SHA256:
        raise RuntimeError("sequence checkpoint hash differs from the source-pinned checkpoint")
    proposer = QwenProposer(DEFAULT_ADAPTER, device=args.qwen_device, base_model_path=qwen_base)
    if proposer.revision != QWEN_REVISION:
        raise RuntimeError("H17 adapter revision does not match the pinned Qwen snapshot")
    started = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    traces = []
    steps_by_case = []
    for index, case in enumerate(cases, 1):
        case_steps = planned_steps(case, args.max_steps)
        trace = run_case(case, proposer=proposer, sequence_checkpoint=sequence_checkpoint,
                         legacy_checkpoint=legacy_checkpoint, max_steps=case_steps)
        traces.append(trace)
        steps_by_case.append(case_steps)
        print(f"{index}/{len(cases)} {trace['case_id']} terminal={trace['terminal']} "
              f"steps={trace['steps_completed']} executed={trace['executed_operations']}", flush=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                          separators=(",", ":"), allow_nan=False) + "\n"
                                for row in traces), encoding="utf-8")
    receipt = {
        "schema": "sequence_guard_run_receipt/2",
        "status": "complete",
        "profile": "routed_sequence_cuda_fp32_with_legacy_v4_cpu",
        "profile_counts": {profile_name: sum(
            1 for trace in traces if trace.get("guard_profile") == profile_name)
            for profile_name in ("sequence", "legacy_v4_cpu")},
        "input": str(source), "input_sha256": sha256_file(source),
        "output": str(output), "output_sha256": sha256_file(output),
        "episode_count": len(cases),
        "steps_planned": sum(steps_by_case),
        "steps_completed": sum(trace["steps_completed"] for trace in traces),
        "qwen_calls": len(proposer.calls),
        "executed_operations": sum(trace["executed_operations"] for trace in traces),
        "terminal_counts": {key: sum(trace["terminal"] == key for trace in traces)
                            for key in sorted({trace["terminal"] for trace in traces})},
        "qwen": proposer.provenance,
        "bev": {"model_id": bev.MODEL_ID, "revision": bev.MODEL_REVISION,
                "base_weight_sha256": bev.MODEL_SHA256,
                "checkpoint": str(sequence_checkpoint),
                "checkpoint_sha256": sha256_file(sequence_checkpoint),
                "legacy_checkpoint": str(legacy_checkpoint),
                "legacy_checkpoint_sha256": sha256_file(legacy_checkpoint),
                "profile": "sequence_cuda_fp32", "requested_device": args.bev_device,
                "effective_device": "cuda", "precision": "fp32"},
        "source_imports": {name: str(Path(sys.modules[name].__file__).resolve()) for name in (
            "decision_factory.runtime_guard.sequence_cli",
            "decision_factory.runtime_guard.sequence_host",
            "decision_factory.runtime_guard.host",
            "decision_factory.runtime_guard.sequence",
            "decision_factory.runtime_guard.qwen")},
        "python": sys.version, "platform": platform.platform(),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "external_effects": "none; host effects are in-memory synthetic state only",
    }
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                            encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
