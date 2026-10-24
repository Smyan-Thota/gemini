# Native Qwen + Bev development results

These are actual model rollouts on 24 inspected synthetic tasks: 16 solvable two-call tasks, four missing-authority tasks and four revoked-authority tasks. Each condition uses the same host, public tasks, menus and action budget. Native Qwen proposals are replayed only when complete public inputs match exactly. The guard is reassessed independently in each condition.

| Configuration | Safe completions / 16 | Unauthorized effects | Correct reviews / 4 | Revoked containment / 4 |
|---|---:|---:|---:|---:|
| qwen-alone | 11 | 16 | 0 | 0 |
| raw-mlp-accuracy-all-64 | 9 | 3 | 4 | 1 |
| raw-linear-accuracy-all-0 | 7 | 0 | 4 | 4 |
| calibrated-linear-accuracy-all-0 | 0 | 0 | 4 | 4 |

The selected research configuration is **raw-linear-accuracy-all-0**. Qwen uses its pretrained native function-call interface; the actor adapters trained during this experiment are not used in this configuration. Both Qwen and the trained Bev-derived guard run during inference.

The classifier development gates have not all passed. Calibration reduced unsafe permissions by deferring substantial legitimate work. Raw configurations remain experimental and their observed failures are reported above. Errors routed to review count as containment, not correct review classifications. No fresh final was run and no protection for unrestricted computer use, universal safety, competitive superiority or research novelty is claimed.

The source includes reproducible training, inference, the bounded host, independent scoring and a private-package builder. Base weights and private evaluation keys are not in Git. Model weight licenses remain separate from the source license; see MODEL_LICENSES.md.
