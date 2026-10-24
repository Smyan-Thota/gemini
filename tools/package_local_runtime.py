"""Create a private portable runtime folder without copying large base models.

The output includes the selected readout and H17 adapter and must stay outside
public Git. Source, dependencies and synthetic demonstration are copied intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output: Path, *, checkpoint: Path, checkpoint_sha256: str, adapter: Path,
          episodes: Path, readout_kind: str = "frozen",
          proposal_mode: str = "finite_mean_logprob", disable_adapter: bool = False,
          factual_progress: bool = False, actor_prompt_profile: str = "legacy",
          result_summary: Path | None = None) -> dict:
    output = output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("private package must be outside the source checkout")
    if output.exists() or output.with_suffix(".zip").exists():
        raise FileExistsError("choose a new package path; existing artifacts are preserved")
    if sha(checkpoint) != checkpoint_sha256.lower():
        raise ValueError("selected readout SHA256 mismatch")
    demo = json.loads(next(line for line in episodes.read_text(encoding="utf-8").splitlines() if line.strip()))
    if (demo.get("schema") != "multi_tool_episode/1"
            or set(demo) - {"schema", "case_id", "request", "action_menu", "max_steps", "max_operations"}):
        raise ValueError("demo must be a public episode without evaluator fields")
    output.mkdir(parents=True)
    roots = {"src": {".py"}, "docs": {".md", ".txt"}, "examples": {".jsonl"}, "tools": {".py"}}
    for directory, suffixes in roots.items():
        for source in sorted((ROOT / directory).rglob("*")):
            if source.is_file() and source.suffix in suffixes and "__pycache__" not in source.parts:
                target = output / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    for name in ("pyproject.toml", "uv.lock", "requirements.txt", "README.md", "LICENSE",
                 "run_multi_tool_demo.ps1", "run_sequence_demo.ps1", "run_sequence_host_demo.ps1"):
        shutil.copyfile(ROOT / name, output / name)
    model_dir = output / "local-models"
    model_dir.mkdir(parents=True)
    if actor_prompt_profile != "native":
        (model_dir / "h17").mkdir()
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copyfile(adapter / name, model_dir / "h17" / name)
    shutil.copyfile(checkpoint, model_dir / "bev-readout.pt")
    (output / "examples" / "multi-tool-demo.jsonl").write_text(
        json.dumps(demo, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    launcher = '''param([Parameter(Mandatory=$true)][string]$QwenBaseModel,[string]$Python="python",[string]$Out="runs/demo.jsonl")
$ErrorActionPreference="Stop"
& (Join-Path $PSScriptRoot "run_multi_tool_demo.ps1") `
  -InputFile (Join-Path $PSScriptRoot "examples/multi-tool-demo.jsonl") `
  -QwenBaseModel $QwenBaseModel ADAPTER_FLAGS `
  -ReadoutCheckpoint (Join-Path $PSScriptRoot "local-models/bev-readout.pt") `
  -ReadoutSha256 "CHECKPOINT_SHA" -ReadoutKind "READOUT_KIND" `
  -QwenProposalMode "PROPOSAL_MODE" -ActorPromptProfile "ACTOR_PROMPT_PROFILE" EXTRA_FLAGS -Python $Python -Out $Out
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
'''.replace("ADAPTER_FLAGS", "" if actor_prompt_profile == "native" else
        '-QwenAdapter (Join-Path $PSScriptRoot "local-models/h17")').replace(
        "CHECKPOINT_SHA", checkpoint_sha256.lower()).replace("READOUT_KIND", readout_kind).replace(
        "PROPOSAL_MODE", proposal_mode).replace("ACTOR_PROMPT_PROFILE", actor_prompt_profile).replace("EXTRA_FLAGS", " ".join(
            flag for flag, enabled in (("-DisableQwenAdapter", disable_adapter),
                                        ("-FactualProgress", factual_progress)) if enabled))
    (output / "run_demo.ps1").write_text(launcher, encoding="utf-8")
    (output / "LOCAL_PACKAGE.md").write_text('''# Local research runtime

This private package includes the selected Bev-derived risk head and an actor adapter
when the selected profile uses one. Native function calling uses base Qwen.
Do not publish its local-models directory. Large pinned base weights are not copied.
Use Python 3.13 and the pinned requirements.txt environment. Bev must already be
available in its pinned local Hugging Face cache or via BEV_MODEL_SNAPSHOT. Qwen's
base snapshot directory must retain the pinned revision name.

Run from this extracted folder:

```powershell
.\\run_demo.ps1 -QwenBaseModel C:\\models\\Qwen\\snapshots\\15852e8c16360a2fea060d615a32b45270f8a8fc
```

All effects are synthetic and remain in memory. The demo is an inspected development
example, not independent final evidence. Result qualification is described in the
source documentation; bundling a model does not promote it or establish universal safety.
''', encoding="utf-8")
    if result_summary is not None:
        shutil.copyfile(result_summary, output / "EXPERIMENT_RESULT.json")
    manifest = {"schema": "local_authorization_package/1", "private_weights_included": True,
                "large_base_weights_included": False, "selected_readout_sha256": checkpoint_sha256.lower(),
                "readout_kind": readout_kind,
                "qwen_proposal_mode": "native_function_call" if actor_prompt_profile == "native" else proposal_mode,
                "qwen_adapter_active": actor_prompt_profile != "native" and not disable_adapter,
                "factual_progress": factual_progress,
                "actor_prompt_profile": actor_prompt_profile,
                "qwen_base_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
                "files": {str(path.relative_to(output)).replace("\\", "/"): sha(path)
                          for path in sorted(output.rglob("*")) if path.is_file()}}
    (output / "BUILD_MANIFEST.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(output.with_suffix(".zip"), "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output))
    return {"folder": str(output), "zip": str(output.with_suffix(".zip")),
            "zip_sha256": sha(output.with_suffix(".zip")), "files": len(manifest["files"])}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--adapter", type=Path, default=ROOT / "adapters" / "h17")
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--readout-kind", choices=("frozen", "adapted"), default="frozen")
    parser.add_argument("--proposal-mode", choices=("finite_mean_logprob", "generate"), default="finite_mean_logprob")
    parser.add_argument("--disable-adapter", action="store_true")
    parser.add_argument("--factual-progress", action="store_true")
    parser.add_argument("--actor-prompt-profile", choices=("legacy", "compact", "native"), default="legacy")
    parser.add_argument("--result-summary", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.output, checkpoint=args.checkpoint,
                           checkpoint_sha256=args.checkpoint_sha256,
                           adapter=args.adapter, episodes=args.episodes, readout_kind=args.readout_kind,
                           proposal_mode=args.proposal_mode, disable_adapter=args.disable_adapter,
                           factual_progress=args.factual_progress,
                           actor_prompt_profile=args.actor_prompt_profile,
                           result_summary=args.result_summary), indent=2))


if __name__ == "__main__":
    main()
