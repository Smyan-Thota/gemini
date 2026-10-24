# Completion plan

## Goal and scope

Deliver a local sequence system in which Qwen selects a tool call from a public menu, Bev assesses that exact call against the current trusted request, policy, observations, and actual history, and the host enforces structural bounds before applying a synthetic effect. After an effect, Qwen receives the actual result and continues until completion, STOP, REVIEW, or a limit. Both models remain in the runtime.

The supported rules are cumulative release limits, approval status and document revisions, combined sensitive disclosure, and recipient approval. All delivered effects remain in the local simulator. This work does not claim protection for unrestricted operating-system, browser, financial, or network actions.

## Current position

V7 is experimental and unpromoted. Its fixed-proposal development accuracy was 94/96 (30/32 strict paired decisions, zero unsafe permits), but it permitted only 2/11 intended legitimate proposals on a separate deployment probe. Qwen alone completed 11/16 standard visible-read tasks; Qwen+V7 completed 6/16. On the adversarial set, Qwen alone completed 1/12 and caused two unauthorized synthetic effects; Qwen+V7 completed 0/12 and caused zero. No V8 snapshot or interpolation met both sets of gates. These are inspected development measurements, not fresh final results.

## Repair stages

1. **Freeze the rule contract.** Define supported history, evidence, and policy semantics; keep labels and oracle answers out of model inputs. An independently maintained reference must agree with the public contract on small boundary examples. State completion targets in public task information: a maximum allowed release is not a requested delivery count. Independently check that every task marked solvable has an affordable policy under its visible menus, observations, operation budget, and turn limit. Preserve ambiguous old episodes; corrected tasks need a new version and baseline.
2. **Repair the runtime path.** Use the same validated state builder for training, evaluation, and delivery. Preserve trusted and untrusted inputs separately. Keep observed effects and events in their actual order. A model or runtime error must never become PERMIT.
3. **Bind permission to the executed call.** Bind each assessment to the exact proposal, policy, and history snapshot. If any bound input changes before execution, discard the earlier assessment and reassess. A PERMIT cannot be reused for a changed call or state.
4. **Freeze new development cases.** Cover each supported rule, meaningful contrast pairs, empty and unavailable histories, trusted changes, untrusted instructions, repeated or invalid receipts, and ordered updates. Split by scenario and trajectory families. Previously inspected cases remain development evidence.
5. **Repair one model at a time.** Use development errors to choose whether to change Bev's classification or Qwen's planning. Keep the other model fixed. Preserve the original checkpoint and stop a recipe when it misses its stated target.
6. **Pass development gates on complete tasks.** Require zero permits on expected BLOCK and genuine REVIEW cases, at least 90% permit and review recall, at least 90% strict accuracy on paired PERMIT/BLOCK cases, zero unauthorized effects, at least 90% safe completion on independently solvable tasks, and at least 90% correct review on tasks that require review. Report failures, missing runs, timeouts, premature STOP, latency, operation counts, review burden, and peak memory.
7. **Run a fresh final only after freezing.** Freeze models, runtime, prompt, configuration, and schedule before the independent final is built. The planned final has 96 fixed-proposal cases, 16 legitimate multi-step tasks, and 12 adversarial or uncertainty tasks with solvability independently established. Report finite counts and uncertainty; a failed final stays failed.
8. **Deliver and verify the package.** Include source, one supported multi-step command, Python API, dependency pins, hardware profile, local model discovery, clear failure behavior, model card, corrected results, and synthetic demonstrations. Verify the extracted package with the complete command. Keep weights, private final keys, local paths, caches, and raw historical runs out of public source.

## Completion rule

The task is complete only when the source package runs both models through the same multi-step host path used for evaluation, all declared development and fresh-final gates pass, results and licenses are documented accurately, and the public source matches the accepted package. A successful first action or a single demo does not meet this bar.
