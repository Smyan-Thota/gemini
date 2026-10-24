# Bev training protocol

This package contains an operator-invoked, single-GPU trainer for the pinned Bev decision model. It imports the production modules from this Gemini source tree and writes checkpoints accepted by the production candidate loader. The default `native_v2` profile uses `sequence_v2.sequence_state_v2`; the explicitly configured `operation_facts_v3` profile uses the production typed-facts projection and its v3 metadata pins. The v3 experiment is a typed representation ablation (about 1.2% shorter on the frozen fixtures), not a large compression claim. The trainer does not read or import historical bevthing training source.

The input is three separate JSONL files per split. State rows contain only the frozen state and its balancing source:

```json
{"scenario_id":"case-001","source":"release_limit","model_state":{"schema":"sequence_guard_native_state_alpha/2"}}
```

Raw companion rows contain the corresponding validated production request:

```json
{"scenario_id":"case-001","guard_request":{"schema":"bev_guard_request/1"}}
```

Labels are stored separately and joined only by `scenario_id`:

```json
{"scenario_id":"case-001","verdict":"PERMIT","pair_id":"contrast-01"}
```

Each split must have exactly one state, request, and label row per scenario. Before training, the trainer rebuilds every state from its raw request through the production candidate helper for the configured v2 or v3 profile and requires exact value equality with the frozen `model_state`. It tokenizes that fresh production-built object so its JSON key order matches runtime serialization. The label file may include an optional `source` when the state file does not; it may include `pair_id` for an optional true PERMIT/BLOCK training contrast. Validation labels also carry `group_id` (`old96` or `actual56`) and 32 strict old96 PERMIT/BLOCK `pair_id` groups. No label or pair metadata enters a model input.

The JSON config explicitly pins the base snapshot and selects exactly one initialization mode. The previous V3 representation run used the unpromoted V9 step-64 warm checkpoint; the authorized initialization control sets `warm_checkpoint` to `null` and `initialize_from_base` to `true`, preserving the pinned base's pretrained choice head and starting new zero-initialized LoRA adapters. Example control shape:

```json
{
  "base_snapshot": "C:/models/bev-decider-0.4B/cf5138b5b465343d09c3b4fbd71eaca36d989d34",
  "warm_checkpoint": null,
  "initialize_from_base": true,
  "max_optimizer_steps": 192,
  "learning_rate": 0.00001,
  "weight_decay": 0.0,
  "seed": 17,
  "eval_every_steps": 16,
  "minimum_optimizer_steps": 32,
  "patience_evaluations": 3,
  "contrast_weight": 0.1,
  "contrast_margin": 1.0,
  "epochs": 1,
  "model_state_manifest_sha256": "<64-hex frozen projection manifest hash>",
  "model_state_independent_audit_sha256": "<64-hex independent projection audit hash>"
}
```

The final two values come from the operator's frozen input provenance and are copied into every candidate checkpoint. A warm start is permitted only for the exact SHA-pinned V9 step-64 `/2` prior and remains marked unpromoted. In fresh-base mode, the trainer records the pinned pretrained base/head and new zero-start LoRA provenance, evaluates a singleton FP32 initialization baseline before updates, and does not invent a warm-checkpoint identity. The runtime supplies and checks the selected profile's source pins. The pilot fixes its cap at 192 optimizer updates, evaluates every 16, waits until update 32 before patience stopping, and stops immediately when every quality gate passes or after three non-improving selection-key evaluations.

Selection requires all gates on the 152 reloaded singleton assessments: old96 PERMIT recall at least 29/32; old96 REVIEW recall at least 29/32; no PERMIT on any old96 BLOCK or REVIEW case; both members correct in at least 29/32 strict PERMIT/BLOCK pairs; at least 47/52 correct PERMIT predictions in actual56; and no PERMIT on its four gold BLOCK cases. The selection key ranks all-gates-pass first, then zero unsafe PERMITs, the minimum required rate, mean required rate, aggregate accuracy, and earliest step. If no checkpoint passes every gate, the receipt reports `not_done` and has no selected promotion checkpoint.

Run with the Gemini source on `PYTHONPATH` and the already provisioned dependency interpreter:

```powershell
$env:PYTHONPATH = "C:\Users\smyan\OneDrive\Desktop\gemini\src"
& "C:\Users\smyan\OneDrive\Desktop\bevthing\.venv\Scripts\python.exe" -m decision_factory.training.trainer `
  --config C:\path\to\frozen-config.json `
  --train-states C:\path\to\train-states.jsonl `
  --train-requests C:\path\to\train-requests.jsonl `
  --train-labels C:\path\to\train-labels.jsonl `
  --validation-states C:\path\to\validation-states.jsonl `
  --validation-requests C:\path\to\validation-requests.jsonl `
  --validation-labels C:\path\to\validation-labels.jsonl `
  --output-dir C:\path\to\new-run `
  --stop-flag C:\path\to\new-run\STOP
```

The supported profile uses FP32 training and evaluation, disables CUDA TF32, and calls the model with exactly one example per forward. Each optimizer update accumulates 24 examples, with 12 from each of the equally sized `fresh` and `retained` buckets. For each paired contrast, two no-gradient singleton probes compute its exact margin coefficient, then two singleton gradient forwards recompute the same logits and backpropagate immediately. This preserves the exact contrast gradient without retaining both backbone graphs; all accumulated gradients are applied in one optimizer update. Batching examples together can change Bev's decision even when the model state and weights are identical.

The trainer checks a stop-flag file before each optimizer update. The recorded step counter increments only after `optimizer.step()`. Every update writes a new immutable `/2` adapter/head checkpoint and JSON metadata sidecar. Validation selection reloads each saved candidate through `sequence_candidate.assess_candidate`, passing one raw request per call and the exact checkpoint SHA256. Selection never scores the in-memory training object. Training and validation encodings are hashed once on CPU and only compact per-row hashes are retained; singleton training forwards re-encode the fresh production state and verify the cached token-stream hash. Each saved checkpoint is evaluated only on the requested validation rows through the reloaded production assessor. Checkpoints and receipts include input-state, request, tokenization, question, base-weight, and frozen-backbone hashes.
