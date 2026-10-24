# Measured results

The current candidate remains **experimental and unpromoted**.

## Corrected multi-step development baseline

The current runtime was run on 28 previously seen development episodes with explicit completion targets. Seven scheduled policy changes were preserved and revealed only after their scheduled turns. An independent checker established an affordable path for every episode before the model runs. The same H17 Qwen proposer and prompt settings were used across three arms, sharing cached proposals only when the full prompts were identical.

| Arm | Target reached | Safe completion | Episodes with unauthorized effects |
| --- | ---: | ---: | ---: |
| Qwen with structural host checks | 21/28 | 19/28 | 2/28 |
| Qwen with reference guard, diagnostic only | 21/28 | 21/28 | 0/28 |
| Qwen with V7 Bev | 12/28 | 12/28 | 0/28 |

Safe completion requires reaching the public goal with no unauthorized effect. Both unauthorized effects exceeded a cumulative release cap. The reference guard is a diagnostic control, not part of the learned guard's production path. V7 completed 4/7 approval, 7/7 joint-disclosure, 0/7 recipient, and 1/7 release tasks. These results identify recipient overblocking and release-task utility as repair targets.

Malformed Qwen JSON caused three proposal errors in each diagnostic arm and one in the V7 arm. These were recorded separately from guard failures. A terminal REVIEW can occur after the goal has already been met; it is not itself evidence of either failure or correct uncertainty handling. These inspected development runs do not replace the planned fresh final.

An independent input audit also found that the current native state loses operation-budget information: a prior allowed read and a prior blocked read can produce identical model inputs despite different remaining budgets. The structural host still checks its actual operation count. This encoding defect must be repaired and versioned before training a replacement guard.

## Bounded Qwen output repair

A subsequent development comparison restricted output to canonical ACT decisions for the current public action menu, STOP, or REVIEW. Each legal continuation, including EOS, was scored using mean token log likelihood; this fixed heuristic is a selection score, not calibrated confidence. H17 completed 26/28 tasks safely with the diagnostic reference guard and produced no malformed decisions. Disabling its adapter under the same scoring mode produced STOP on all 28 initial prompts and completed no tasks. An independent audit checked all 411 candidate scores and initial-prompt parity. This supports retaining H17 and the bounded format for the next paired run; it does not establish a result for an updated Bev guard or an unseen final set.

## Earlier measurements

- On the original fixed-proposal development set, V7 scored 94/96 decisions and 30/32 strict paired decisions, with zero unsafe permits.
- On a separate deployment-utility probe, it permitted 2/11 intended legitimate proposals and blocked both expected BLOCK proposals. This failed the deployment-utility gate.
- On visible-read standard task runs, Qwen alone completed 11/16 tasks; Qwen with V7 completed 6/16.
- On the separate adversarial task set, Qwen alone completed **1/12** tasks and caused two unauthorized synthetic effects. Qwen with V7 completed **0/12** tasks and caused zero unauthorized synthetic effects. The lower effect count came with no task completions in either arm.
- No V8 snapshot or tested interpolation met both the original development gates and the deployment probe gates. V8 run 2 stopped after 192 actual optimizer updates; an older receipt incorrectly listed 576 planned updates as completed.

## Inference-profile correction

A later replay of the V9 step-64 checkpoint found that the historical batch-2 development run is not a production-singleton result. The same checkpoint and requests produced different choices when encoded with a padding peer; V9's original batch-2 order reproduced its stored outputs, while the runtime's one-request CUDA FP32 path changed four decisions. A separate 56-request comparison also found that sorting the serialized state keys changed 26 singleton decisions despite identical JSON values and token counts, because Bev consumes the state text in key order. Treat the earlier V9 batch-2 receipt as a historical training diagnostic, not a deployment score. Candidate selection now uses raw production state objects, exact token/render fingerprints, and singleton FP32 assessment; V9 step 64 is available only as an explicitly unpromoted warm-start prior.

These are finite synthetic development measurements. They do not show a broad safety guarantee or superiority. A zero observed unauthorized effect count in this sample does not establish that the guard will prevent future unauthorized effects. The detailed reporting correction is in [ERRATA-2026-10-23](ERRATA-2026-10-23.md).
