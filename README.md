# Gemini: Qwen + Bev sequence guard

This repository contains the source for a local, experimental sequence runtime guard. Qwen proposes a tool call, Bev assesses the exact proposal using the current task policy and history, and a bounded host applies only synthetic in-memory effects. The included examples contain no evaluator labels. The host demo performs no filesystem, network, or account actions.

The current V7 Bev guard is **experimental and unpromoted**. It reduced observed unauthorized effects in one small adversarial set, but it also reduced completion on standard tasks and completed none of the 12 adversarial tasks. These results do not establish general safety or superiority. See [measured results](docs/RESULTS.md) and the [completion plan](docs/COMPLETION_PLAN.md).

## Source-only checkout

The initial public snapshot contains source, pinned dependencies, and label-free examples. It does not contain Qwen adapters, Bev checkpoints, base-model weights, evaluation keys, or local caches. Runtime commands therefore need the matching local model files before they can run.

The package targets Python 3.13 and pins its optional inference stack in `pyproject.toml`, `requirements.txt`, and `uv.lock`. The active GPU profile uses CUDA 12.8 and FP32 for Bev. Qwen uses the existing `Qwen/Qwen3.5-2B` snapshot at revision `15852e8c16360a2fea060d615a32b45270f8a8fc`; Bev uses `avbiswas/bev-decider-0.4B` at revision `cf5138b5b465343d09c3b4fbd71eaca36d989d34`. Weight licenses and non-commercial restrictions are summarized in [model licenses](docs/MODEL_LICENSES.md).

The source expects the local H17 adapter and V7/legacy Bev checkpoint files under `adapters/h17/` and `checkpoints/`. Keep these files out of version control. Base weights must already be cached locally; the runtime is configured for offline loading. Run `run_sequence_demo.ps1` or `run_sequence_host_demo.ps1` for the bounded multi-step Qwen → Bev → in-memory host loop. The equivalent command is:

```powershell
python -m decision_factory.runtime_guard.sequence_cli `
  --input examples/sequence-host-demo.jsonl `
  --out runs/sequence-host-demo.jsonl `
  --qwen-base C:\path\to\Qwen\snapshots\15852e8c16360a2fea060d615a32b45270f8a8fc
```

The sequence Bev profile requires CUDA FP32. `--qwen-device cpu` is supported for the proposer if GPU memory is constrained; `--bev-device cpu` is intentionally unsupported. The prior one-turn assessment-only command is retained as `sequence-guard-assess-legacy` (`sequence-guard-demo` is its compatibility alias). The full host command is `sequence-guard` (`sequence-guard-host-demo` is its alias). These demos execute only synthetic in-memory effects and never access the operating system or network.

## Scope

The supported workflow rule types are cumulative release limits, approval status, joint disclosure, and recipient allowlists. Unknown or unsupported rule combinations fail closed. Structural checks for schemas, allowed operations, workspace boundaries, and operation budgets remain host responsibilities. This prototype does not mediate real operating-system, browser, financial, or network actions.

The exact development and deployment gates, repair stages, and completion conditions are in [the completion plan](docs/COMPLETION_PLAN.md). Historical private runs and test keys are not part of this repository.
