"""Run base Qwen native calls against the public 24-episode Bev comparison."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QWEN = (Path(os.environ["USERPROFILE"]) / ".cache" / "huggingface" / "hub" /
                "models--Qwen--Qwen3.5-2B" / "snapshots" /
                "15852e8c16360a2fea060d615a32b45270f8a8fc")
EXPECTED_ARMS = ("qwen-alone", "raw-mlp-accuracy-all-64",
                 "raw-linear-accuracy-all-0", "calibrated-linear-accuracy-all-0")


class ExactInputReplay:
    """Replay only identical public inputs across deterministic comparison arms."""

    def __init__(self, real):
        self.real, self.cache, self.calls = real, {}, []

    @property
    def provenance(self):
        return {**self.real.provenance, "exact_public_input_replay": True}

    def propose(self, public):
        key = hashlib.sha256(canonical(public).encode()).hexdigest()
        replayed = key in self.cache
        if not replayed:
            before = len(self.real.calls)
            result = self.real.propose(public)
            audit = self.real.calls[-1] if len(self.real.calls) > before else {"model_call": False}
            self.cache[key] = copy.deepcopy((result, audit))
        result, audit = copy.deepcopy(self.cache[key])
        self.calls.append({**audit, "public_input_sha256": key, "replayed": replayed})
        result["exact_public_input_replayed"] = replayed
        return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def durable_json(path: Path, value: Any, *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                allow_nan=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def selected_artifacts(comparison_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    candidates = comparison.get("candidates")
    if not isinstance(candidates, dict):
        raise ValueError("comparison.json has no candidate map")
    specs = {
        "raw-mlp-accuracy-all-64": ("accuracy-all-64", "raw"),
        "raw-linear-accuracy-all-0": ("accuracy-all-0", "raw"),
        "calibrated-linear-accuracy-all-0": ("accuracy-all-0", "calibrated"),
    }
    artifacts = {}
    for arm, (candidate_name, variant) in specs.items():
        candidate = candidates.get(candidate_name)
        if not isinstance(candidate, dict):
            raise ValueError(f"comparison.json lacks {candidate_name}")
        path_key, hash_key = (("raw_path", "raw_sha256") if variant == "raw"
                              else ("calibrated_path", "calibrated_sha256"))
        path, expected = Path(candidate[path_key]).expanduser().resolve(strict=True), candidate[hash_key]
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{candidate_name} {variant} checkpoint hash differs from comparison.json")
        artifacts[arm] = {"path": path, "sha256": expected, "candidate": candidate_name,
                          "variant": variant}
    return artifacts, comparison


def read_public_episodes(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 24:
        raise ValueError(f"expected exactly 24 public episodes, got {len(rows)}")
    if any(row.get("schema") != "multi_tool_episode/1" for row in rows):
        raise ValueError("input contains a non-public or unsupported episode")
    return rows


def readout_identity(readout: Any, artifact: dict[str, Any]) -> dict[str, Any]:
    payload = readout.payload
    return {"path": str(artifact["path"]), "sha256": artifact["sha256"],
            "comparison_candidate": artifact["candidate"], "variant": artifact["variant"],
            "schema": payload.get("schema"), "feature_profile": payload.get("feature_profile"),
            "base_revision": payload.get("base_revision"),
            "base_weight_sha256": payload.get("base_weight_sha256"),
            "adaptation_source_sha256": payload.get("adaptation_source_sha256"),
            "adapter_state_sha256": payload.get("adapter_state_sha256"),
            "layers": len(payload.get("layers", [])),
            "feature_columns": len(payload.get("columns", [])),
            "calibrated": bool(payload.get("calibrated")),
            "temperature": float(payload.get("temperature", 1.0)),
            "permit_threshold": float(payload.get("permit_threshold", 0.0)),
            "block_threshold": float(payload.get("block_threshold", 0.0))}


def run_arm(arm: str, cases: list[dict[str, Any]], proposer: Any,
            assessor: Any, out_dir: Path, *, input_path: Path,
            comparison_path: Path, comparison_sha256: str,
            model_provenance: dict[str, Any], source_hashes: dict[str, str],
            artifact: dict[str, Any] | None) -> dict[str, Any]:
    from decision_factory.runtime_guard.multi_tool_host import run_multi_tool_episode

    trace_path = out_dir / f"{arm}.traces.jsonl"
    started = time.perf_counter()
    with trace_path.open("x", encoding="utf-8", newline="\n") as stream:
        for index, case in enumerate(cases, 1):
            call_start = len(proposer.calls)
            trace = run_multi_tool_episode(case, proposer=proposer,
                assessor=assessor.assess_scene if assessor is not None else None)
            trace["arm"] = arm
            trace["qwen_model_provenance"] = model_provenance
            trace["proposal_calls"] = proposer.calls[call_start:]
            trace["input_sha256"] = sha256_file(input_path)
            trace["comparison_sha256"] = comparison_sha256
            if artifact is not None:
                trace["bev_artifact_sha256"] = artifact["sha256"]
            stream.write(json.dumps(trace, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            print(f"{arm} {index}/{len(cases)} {trace['case_id']} {trace['terminal']} "
                  f"effects={trace['executed_operations']}", flush=True)
    receipt = {"schema": "native_guard_comparison_arm_receipt/1", "status": "complete",
               "arm": arm, "episodes": len(cases), "trace_path": str(trace_path),
               "trace_sha256": sha256_file(trace_path), "input_path": str(input_path),
               "input_sha256": sha256_file(input_path),
               "comparison_path": str(comparison_path), "comparison_sha256": comparison_sha256,
               "qwen": model_provenance, "bev": artifact, "source_hashes": source_hashes,
               "python": sys.version, "platform": platform.platform(),
               "wall_seconds": round(time.perf_counter() - started, 3),
               "external_effects": "none; synthetic ledger only"}
    durable_json(out_dir / f"{arm}.receipt.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--qwen-base", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--limit", type=int, help="partial development smoke, never the full comparison")
    args = parser.parse_args(argv)

    input_path = args.input.expanduser().resolve(strict=True)
    comparison_path = args.comparison.expanduser().resolve(strict=True)
    qwen_base = args.qwen_base.expanduser().resolve(strict=True)
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists():
        raise FileExistsError("choose a new output directory; previous runs are preserved")
    cases = read_public_episodes(input_path)
    if args.limit is not None:
        if not 1 <= args.limit <= len(cases):
            raise ValueError("limit must be between 1 and 24")
        cases = cases[:args.limit]
    artifacts, comparison = selected_artifacts(comparison_path)
    if comparison.get("scope") != "DEV/CALonly;frozenbackbone;finalcorrectionround2":
        raise ValueError("comparison.json is not the expected producer-readout round 2 record")
    comparison_sha256 = sha256_file(comparison_path)
    source_paths = {name: ROOT / "src" / "decision_factory" / "runtime_guard" / name
                    for name in ("native_actor.py", "multi_tool_host.py", "adapted_readout.py",
                                 "frozen_readout.py", "bev.py")}
    source_paths["runner"] = Path(__file__)
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}

    sys.path.insert(0, str(ROOT / "src"))
    from decision_factory.runtime_guard.native_actor import NativeQwenProposer
    if qwen_base.name != "15852e8c16360a2fea060d615a32b45270f8a8fc":
        raise ValueError("Qwen base snapshot revision mismatch")
    out_dir.mkdir(parents=True)
    proposer = ExactInputReplay(NativeQwenProposer(qwen_base, device="cuda", max_new_tokens=128))
    model_provenance = proposer.provenance
    receipts = []
    for arm in EXPECTED_ARMS:
        assessor = artifact = None
        if arm == "qwen-alone":
            receipts.append(run_arm(arm, cases, proposer, assessor, out_dir,
                input_path=input_path, comparison_path=comparison_path,
                comparison_sha256=comparison_sha256, model_provenance=model_provenance,
                source_hashes=source_hashes, artifact=None))
            continue
        artifact = artifacts[arm]
        from decision_factory.runtime_guard.adapted_readout import load_adapted_readout
        assessor = load_adapted_readout(artifact["path"], artifact["sha256"], device="cuda")
        identity = readout_identity(assessor, artifact)
        # Record the arm-specific loaded head identity in traces and receipt.
        receipts.append(run_arm(arm, cases, proposer, assessor, out_dir,
            input_path=input_path, comparison_path=comparison_path,
            comparison_sha256=comparison_sha256, model_provenance=model_provenance,
            source_hashes=source_hashes, artifact=identity))
        del assessor
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    durable_json(out_dir / "run.receipt.json", {
        "schema": "native_guard_comparison_receipt/1", "status": "complete",
        "arms": list(EXPECTED_ARMS), "episodes_per_arm": len(cases),
        "input_path": str(input_path), "input_sha256": sha256_file(input_path),
        "comparison_path": str(comparison_path), "comparison_sha256": comparison_sha256,
        "qwen": model_provenance, "arm_receipts": receipts,
        "source_hashes": source_hashes,
        "external_effects": "none; synthetic ledger only"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
