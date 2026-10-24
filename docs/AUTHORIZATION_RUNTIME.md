# Authorization across tools

The new experimental path uses the same local Qwen actor with a specialized readout over frozen Bev features. Qwen selects an exact call from a public action menu. Bev evaluates the current request, policy, prior actions, observations and pending call. The host commits a simulated effect only after PERMIT. Both models remain in inference.

This path accepts uniform synthetic tool contracts with target, destination and integer-unit arguments. Tool names may represent file, messaging, records, spending or access actions. The executor records these effects in memory; it does not perform the corresponding real-world operations. A tool name is not evidence that the system protects that real tool.

## Run

The native actor profile uses Qwen's own function-call template and the trained
Bev guard. It does not load a Qwen adapter or select artificial action IDs:

```powershell
python -m decision_factory.runtime_guard.multi_tool_cli --actor-prompt-profile native --mode matched --qwen-base C:\models\Qwen\snapshots\15852e8c16360a2fea060d615a32b45270f8a8fc --input examples\multi-tool-demo.jsonl --readout-kind adapted --readout-checkpoint C:\models\bev-readout.pt --readout-sha256 <sha256> --out runs\native-demo.jsonl
```

Native proposals still pass through the same exact-call checks, Bev assessment,
and synthetic effect ledger. Unknown native output requests review and is
reported separately from a correctly classified review. The model weights and
qualified results must be supplied with the selected local package.

Install the pinned optional model dependencies and provide matching local model files. The readout loader pins Bev; the Qwen adapter identifies its required base snapshot. The command makes no implicit model download. Keep weights outside the public checkout.

```powershell
.\run_multi_tool_demo.ps1 `
  -InputFile C:\models\demo\episodes.jsonl `
  -QwenBaseModel C:\models\Qwen\snapshots\15852e8c16360a2fea060d615a32b45270f8a8fc `
  -QwenAdapter C:\models\h17 `
  -ReadoutCheckpoint C:\models\bev-readout.pt `
  -ReadoutSha256 <checkpoint-sha256> `
  -Out runs\multi-tool-demo.jsonl
```

The equivalent Python entrypoint is `python -m decision_factory.runtime_guard.multi_tool_cli`. CPU placement is accepted for either model, with lower throughput. A completed run writes one trace per episode and a sibling receipt. Existing outputs are never overwritten.

Pass `-Mode matched` to run Qwen alone and Qwen plus Bev with the same actor, menus, hard host checks and attempt budget. Each condition starts from a separate copy of the episode. These are actual model rollouts; private scoring keys are not passed to either model. The Python command also supports `--mode qwen-alone` without loading a readout.

## Public input

Each JSONL row has schema `multi_tool_episode/1` and contains `case_id`, `request`, `action_menu`, `max_steps` and `max_operations`. The request uses `bev_guard_request/1`. Each menu item contains `action_id`, `description`, `operation` and `arguments`. Evaluator labels, winning actions and private keys must be in a separate scoring file. The runtime rejects extra episode fields.

The request is host-authored. Only the trusted application may assign trust labels or author policy. Untrusted document text remains untrusted even if it claims to be a user approval. The caller must not expose this loader to an untrusted party as an authority-creation interface.

## Intervention and bounds

Every call is checked afresh. The assessed scene is a detached copy, and the host binds the exact proposal and state before commit. A malformed result or assessor failure requests review without effects. A blocked proposal gets a bounded recovery opportunity; proposing the same blocked call twice ends the episode. Each retry is independently assessed. Review terminates the episode and is deferred work, not successful completion.

Only hard schema, menu and budget checks are implemented by this host. It does not contain the training reference evaluator or compute authorization labels. A no-learned-guard comparison uses the same hard checks. Permission is not task correctness. Reaching an operation budget is not evidence that the task succeeded; independent scoring must inspect the committed effects against the private goal and authorization conditions.

## Evidence status

This is a research interface. Classifier accuracy, host unit checks and complete Qwen-plus-Bev task outcomes are different evidence and must be reported separately. Training on synthetic contracts does not establish protection for unrestricted computer use. New readout results must be reported with their data split, actual counts, block/review burden and runtime profile. Existing failed experiments remain preserved.
