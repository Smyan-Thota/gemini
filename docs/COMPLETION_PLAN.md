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

## Corrected training pilot

The next pilot fixes two measured input differences: padding across examples changed Bev decisions, and JSON key order changed the token stream even when state values matched. Training must therefore rebuild the production state from each raw request and use the same singleton FP32 encoding as inference. Stored projections are audit records, not training token inputs.

The frozen training mix has 4,608 examples: 2,304 prior repair examples, 1,536 new hard cases, and 768 mixed-tool READ cases. Two equally sized source buckets keep new examples and prior coverage balanced. Sequential singleton forwards with gradient accumulation over 24 examples allow the 192-update cap to cover the complete set once. Evaluation reloads saved weights through the production assessor every 16 optimizer updates. It stops on success, or after three evaluations without improvement after update 32. The historical warm checkpoint is an unpromoted starting point.

Selection uses 96 fixed proposals and 56 proposals from actual development trajectories. It requires zero unsafe permits; at least 29/32 correct PERMIT decisions, 29/32 correct REVIEW decisions, and 29/32 fully correct PERMIT/BLOCK pairs on the fixed set; and at least 47/52 correct PERMIT decisions on the trajectory set, with no PERMIT on its four BLOCK cases. These are development gates. Passing them triggers complete task runs, followed by the independently generated final set after the full system is frozen.

## Historical matched representation experiment

The corrected pilot completed 192 updates, but no checkpoint met all gates. Its highest aggregate score was 126/152 with six unsafe permits. A checkpoint with zero unsafe permits scored 85/152 and blocked many valid actions. Errors occurred across all four rule families; overall training labels and source counts were balanced.

The representation experiment changed the model input representation while retaining the same 4,608 training requests, 152 development requests, labels, initial optimization settings, and acceptance gates. It tested whether compact typed facts make the relevant policy, proposal, and committed history easier to distinguish than repeated prose and opaque identifiers. Both warm-start and fresh-base controls completed 192 updates without passing the combined gates. Their artifacts remain preserved; the unchanged recipe is not scheduled again.

An independent feature audit must precede implementation. The projection must preserve missing versus empty evidence, trusted event order, actual versus attempted effects, and exact policy relationships. Arithmetic and membership facts may be exposed; computed decisions or violation flags may not. The runtime projector must not import the reference evaluator or receive labels, future events, or test keys. The existing input profile remains available unchanged. Training and live assessment must use one shared encoder for the new profile, with an actual model parity and memory check before the full run.

## Resumed research direction: frozen features and authorization across tools

The next method trains a small risk readout over frozen Bev representations. First measure this changed method on the existing train/development inputs; then use grouped, matched authorized/unauthorized/insufficient-evidence examples across file, messaging, records and spending/access tool contracts. Closely related branches remain in one split. Tool-family transfer and novel scenario structures are reported separately from ordinary held-out rows. No previously inspected set becomes a fresh final.

The runtime remains Qwen -> Bev-derived guard -> bounded synthetic host. The new [multi-tool interface](AUTHORIZATION_RUNTIME.md) separates the learned permission decision from hard schema/menu/budget checks. Missing evidence and uncertain decisions request review. Blocked calls may lead to a different proposal; repeated blocked calls and total attempts are bounded. Reviews are deferred tasks, not completions.

Frozen features are a hypothesis, not a claim that large-model research transfers to Bev. At most one limited-adaptation fallback and two targeted correction rounds are allowed before a method decision. No gate is lowered. Complete task runs with the same actor, host and recovery budget must establish an improvement in unsafe effects and useful completion before promotion. Existing full-task and fresh-final completion requirements above remain in force.
