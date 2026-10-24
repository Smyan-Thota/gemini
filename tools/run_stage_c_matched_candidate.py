"""Run the frozen 28-episode Stage C suite with shared finite-H17 proposals.

This is an operator tool, not part of the runtime package. It writes only local
JSONL traces and receipts. All simulated effects stay in GuardHost's memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path(os.environ["LOCALAPPDATA"]) / "gemini" / "stage-c-v2" / "episodes.jsonl"
DEFAULT_QWEN_BASE = (Path(os.environ["USERPROFILE"]) / ".cache" / "huggingface" / "hub" /
                     "models--Qwen--Qwen3.5-2B" / "snapshots" /
                     "15852e8c16360a2fea060d615a32b45270f8a8fc")
DEFAULT_ADAPTER = ROOT / "adapters" / "h17"
DEFAULT_V7 = ROOT / "checkpoints" / "sequence-bev-v7-experimental.pt"
DEFAULT_LEGACY = ROOT / "checkpoints" / "legacy-v4-bev.pt"
INPUT_SHA256 = "198cf0560ab8dce1d09d5a88d92c42aff5378687c03c373f052068c09d2065f2"
STAGE_C_MANIFEST_SHA256 = "4331bd6eceafb35cc385fe4289f0afadfd4dec4b2e4c15fbd7e106530cb36251"
QWEN_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
V7_CHECKPOINT_SHA256 = "57492578fee9ca6cc1d31b0c1a18900e95581d670ab7b4d1a6322b7b88523edf"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


class SharedFiniteH17:
    """Use production H17 rendering/scoring and replay only exact same prompts."""

    def __init__(self, real: Any):
        if real.proposal_mode != "finite_mean_logprob":
            raise ValueError("matched Stage C requires finite_mean_logprob")
        self.real = real
        self.cache: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        self.calls: list[dict[str, Any]] = []
        self.cache_hits = 0
        self.provenance = {**real.provenance,
                           "matched_proposal_mode": "finite_mean_logprob",
                           "exact_prompt_replay": True,
                           "cache_key": "sha256 of exact production-rendered prompt"}

    def propose(self, task_input: dict[str, Any]) -> dict[str, Any]:
        # Ask the production renderer for its exact prompt while replacing only
        # the callback for this render-only call. No model output is consulted.
        captured: list[str] = []
        old_callback, old_mode = self.real.callback, self.real.proposal_mode
        prior_call_count = len(self.real.calls)
        self.real.callback = lambda prompt: (captured.append(prompt) or '{"type":"STOP"}')
        self.real.proposal_mode = "generate"
        try:
            self.real.propose(task_input)
        finally:
            self.real.callback = old_callback
            self.real.proposal_mode = old_mode
            del self.real.calls[prior_call_count:]
        if len(captured) != 1:
            raise RuntimeError("production H17 renderer did not yield exactly one prompt")
        prompt = captured[0]
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if prompt_sha in self.cache:
            result, stored_call = self.cache[prompt_sha]
            self.cache_hits += 1
            self.calls.append({**stored_call, "replayed": True})
            return json.loads(json.dumps(result))

        from decision_factory.recovery.computer_safety import ToolAction
        actions = [ToolAction(row["action_id"], row["description"],
                              row.get("operation", row.get("tool_name")),
                              dict(row["arguments"]))
                   for row in task_input["action_menu"]]
        result = self.real._finite_menu_decision(prompt, actions)
        call = {"prompt": prompt, "prompt_sha256": prompt_sha,
                "generation_audit": result["generation_audit"], "replayed": False}
        self.cache[prompt_sha] = (json.loads(json.dumps(result)), call)
        self.calls.append(call)
        return result


def run_arm(arm: str, cases: list[dict[str, Any]], *, proposer: SharedFiniteH17,
            v7_checkpoint: Path, legacy_checkpoint: Path,
            candidate_checkpoint: Path, candidate_sha256: str) -> list[dict[str, Any]]:
    from decision_factory.runtime_guard.host import GuardHost
    from decision_factory.runtime_guard.sequence_cli import planned_steps
    from decision_factory.runtime_guard.sequence_host import run_sequence_episode

    traces = []
    for index, case in enumerate(cases, 1):
        base = {"schema": "bev_guard_request/1", "user_request": case["task"],
                "host_policy": case["host_policy"],
                "prior_actions": case.get("prior_actions", []),
                "observations": case.get("initial_observations", []), "proposal": {}}
        if arm == "qwen_alone":
            host = GuardHost(request=base, initial_files=case["virtual_files"],
                             mode="qwen-alone")
        else:
            if arm == "v7":
                from decision_factory.runtime_guard.sequence import evaluate_routed_proposal

                def assessor(request: dict[str, Any], *, device: str = "cuda",
                             head_checkpoint: str | Path | None = None) -> dict[str, Any]:
                    return evaluate_routed_proposal(
                        request, sequence_checkpoint=head_checkpoint or v7_checkpoint,
                        legacy_checkpoint=legacy_checkpoint, sequence_device=device)
            elif arm == "candidate":
                from decision_factory.runtime_guard.sequence_candidate import assess_candidate

                def assessor(request: dict[str, Any], *, device: str = "cuda",
                             head_checkpoint: str | Path | None = None) -> dict[str, Any]:
                    if "workflow_rules" not in request.get("host_policy", {}):
                        from decision_factory.runtime_guard.sequence import evaluate_routed_proposal
                        return evaluate_routed_proposal(
                            request, sequence_checkpoint=v7_checkpoint,
                            legacy_checkpoint=legacy_checkpoint, sequence_device=device)
                    return assess_candidate(
                        request, checkpoint_path=head_checkpoint or candidate_checkpoint,
                        expected_checkpoint_sha256=candidate_sha256, device=device)
            else:
                raise ValueError(f"unknown arm: {arm}")
            legacy_route = "workflow_rules" not in case["host_policy"]
            host = GuardHost(
                request=base, initial_files=case["virtual_files"], mode="qwen-bev",
                bev_evaluator=assessor,
                bev_head_checkpoint=str(candidate_checkpoint if arm == "candidate" else v7_checkpoint),
                bev_device="cpu" if legacy_route else "cuda",
                profile="legacy_v4_cpu" if legacy_route else "sequence")

        trace = run_sequence_episode(case, proposer=proposer, host=host,
                                     max_steps=planned_steps(case, None))
        trace["arm"] = arm
        trace["qwen_provenance"] = proposer.provenance
        trace["stage_c_manifest_sha256"] = STAGE_C_MANIFEST_SHA256
        traces.append(trace)
        print(f"{arm} {index}/{len(cases)} {trace['case_id']} "
              f"terminal={trace['terminal']} steps={trace['steps_completed']} "
              f"executed={trace['executed_operations']} prompt_cache_hits={proposer.cache_hits}",
              flush=True)
    return traces


def write_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--input-sha256", default=INPUT_SHA256)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--qwen-base", type=Path, default=DEFAULT_QWEN_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--v7-checkpoint", type=Path, default=DEFAULT_V7)
    parser.add_argument("--legacy-checkpoint", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true",
                        help="verify frozen inputs and local model/checkpoint identities without loading models")
    args = parser.parse_args(argv)

    input_path = args.input.expanduser().resolve(strict=True)
    out_dir = args.out_dir.expanduser().resolve()
    qwen_base = args.qwen_base.expanduser().resolve(strict=True)
    adapter = args.adapter.expanduser().resolve(strict=True)
    v7_checkpoint = args.v7_checkpoint.expanduser().resolve(strict=True)
    legacy_checkpoint = args.legacy_checkpoint.expanduser().resolve(strict=True)
    candidate_checkpoint = args.candidate_checkpoint.expanduser().resolve(strict=True)
    if sha256_file(input_path) != args.input_sha256.lower():
        raise ValueError("Stage C episode bytes differ from the frozen expected SHA256")
    if sha256_file(v7_checkpoint) != V7_CHECKPOINT_SHA256:
        raise ValueError("V7 checkpoint differs from the reviewed baseline artifact")
    if sha256_file(candidate_checkpoint) != args.candidate_sha256.lower():
        raise ValueError("candidate checkpoint does not match the operator SHA256")
    if qwen_base.name != QWEN_REVISION:
        raise ValueError("Qwen base snapshot revision mismatch")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("choose a new empty output directory; runs are immutable")

    sys.path.insert(0, str(ROOT / "src"))
    from decision_factory.runtime_guard.sequence_cli import read_cases
    from decision_factory.runtime_guard.qwen import QwenProposer
    from decision_factory.runtime_guard.sequence_candidate import release_candidate_cache
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("matched sequence run requires CUDA for Bev")
    cases = read_cases(input_path)
    if len(cases) != 28:
        raise ValueError(f"expected 28 frozen Stage C episodes, got {len(cases)}")
    manifest_path = input_path.parent / "manifest.json"
    if sha256_file(manifest_path) != STAGE_C_MANIFEST_SHA256:
        raise ValueError("Stage C manifest differs from the frozen manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = {"scheduled": sum(bool(case.get("environment_schedule")) for case in cases),
              "standard": manifest.get("standard_count"),
              "adversarial": manifest.get("adversarial_count")}
    if (manifest.get("case_count") != len(cases)
            or counts != {"scheduled": 7, "standard": 16, "adversarial": 12}):
        raise ValueError(f"Stage C cohort structure mismatch: {counts}")
    if args.preflight_only:
        print(json.dumps({"status": "preflight_pass", "input": str(input_path),
                          "input_sha256": sha256_file(input_path),
                          "manifest": str(manifest_path),
                          "manifest_sha256": sha256_file(manifest_path),
                          "episodes": len(cases), "cohort_counts": counts,
                          "qwen_revision": QWEN_REVISION,
                          "v7_checkpoint_sha256": sha256_file(v7_checkpoint),
                          "candidate_checkpoint_sha256": sha256_file(candidate_checkpoint),
                          "legacy_checkpoint_sha256": sha256_file(legacy_checkpoint)},
                         ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    real = QwenProposer(adapter, device="cuda", base_model_path=qwen_base,
                        proposal_mode="finite_mean_logprob")
    if real.revision != QWEN_REVISION:
        raise RuntimeError("H17 adapter revision mismatch")
    proposer = SharedFiniteH17(real)
    started = time.perf_counter()
    outputs: dict[str, list[dict[str, Any]]] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError("choose a new empty output directory; runs are immutable")
    for arm in ("qwen_alone", "v7", "candidate"):
        outputs[arm] = run_arm(
            arm, cases, proposer=proposer, v7_checkpoint=v7_checkpoint,
            legacy_checkpoint=legacy_checkpoint, candidate_checkpoint=candidate_checkpoint,
            candidate_sha256=args.candidate_sha256.lower())
        path = out_dir / f"{arm}.traces.jsonl"
        write_exclusive(path, "".join(canonical(trace) + "\n" for trace in outputs[arm]))
        if arm == "v7":
            # Avoid retaining V7's GPU model while loading the candidate. This
            # is the public cache API; preserve model files and checkpoint bytes.
            release_candidate_cache(
                v7_checkpoint, expected_checkpoint_sha256=sha256_file(v7_checkpoint))
            torch.cuda.empty_cache()

    from decision_factory.runtime_guard import bev
    from decision_factory.runtime_guard import sequence_candidate, sequence_cli, sequence_host
    from decision_factory.runtime_guard import sequence as sequence_runtime
    from decision_factory.runtime_guard import host as host_runtime, qwen as qwen_runtime
    arm_receipts = {}
    for arm, traces in outputs.items():
        path = out_dir / f"{arm}.traces.jsonl"
        arm_receipts[arm] = {
            "path": str(path), "sha256": sha256_file(path), "episodes": len(traces),
            "steps": sum(row["steps_completed"] for row in traces),
            "executed_operations": sum(row["executed_operations"] for row in traces),
            "terminal_counts": {key: sum(row["terminal"] == key for row in traces)
                                for key in sorted({row["terminal"] for row in traces})},
        }
    receipt = {
        "schema": "stage_c_matched_finite_candidate_run/1", "status": "complete",
        "input": str(input_path), "input_sha256": sha256_file(input_path),
        "stage_c_manifest_sha256": STAGE_C_MANIFEST_SHA256,
        "stage_c_manifest_path": str(manifest_path),
        "stage_c_manifest_file_sha256": sha256_file(manifest_path),
        "qwen": real.provenance,
        "proposal_matching": {
            "method": "exact production-rendered prompt SHA256 cache across arms",
            "cache_hits": proposer.cache_hits, "unique_prompts": len(proposer.cache),
            "actual_finite_scoring_calls": len(proposer.cache),
            "rule": "mean causal completion-token log-likelihood through canonical JSON and EOS",
            "tie_break": "REVIEW, STOP, then ACT in public menu order",
        },
        "arms": arm_receipts,
        "v7_checkpoint_sha256": sha256_file(v7_checkpoint),
        "candidate_checkpoint": str(candidate_checkpoint),
        "candidate_checkpoint_sha256": sha256_file(candidate_checkpoint),
        "legacy_checkpoint_sha256": sha256_file(legacy_checkpoint),
        "external_effects": "none; GuardHost effects are synthetic in-memory only",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "python": sys.version, "platform": platform.platform(),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "runtime_source_sha256": {
            module.__name__: sha256_file(Path(module.__file__).resolve())
            for module in (bev, sequence_candidate, sequence_cli, sequence_host,
                           sequence_runtime, host_runtime, qwen_runtime)
        },
    }
    receipt_path = out_dir / "receipt.json"
    write_exclusive(receipt_path, json.dumps(receipt, ensure_ascii=False,
                                             sort_keys=True, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
