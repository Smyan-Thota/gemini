# Measured results

The current candidate remains **experimental and unpromoted**.

- On the original fixed-proposal development set, V7 scored 94/96 decisions and 30/32 strict paired decisions, with zero unsafe permits.
- On a separate deployment-utility probe, it permitted 2/11 intended legitimate proposals and blocked both expected BLOCK proposals. This failed the deployment-utility gate.
- On visible-read standard task runs, Qwen alone completed 11/16 tasks; Qwen with V7 completed 6/16.
- On the separate adversarial task set, Qwen alone completed **1/12** tasks and caused two unauthorized synthetic effects. Qwen with V7 completed **0/12** tasks and caused zero unauthorized synthetic effects. The lower effect count came with no task completions in either arm.
- No V8 snapshot or tested interpolation met both the original development gates and the deployment probe gates. V8 run 2 stopped after 192 actual optimizer updates; an older receipt incorrectly listed 576 planned updates as completed.

These are finite synthetic development measurements. They do not show a broad safety guarantee or superiority. A zero observed unauthorized effect count in this sample does not establish that the guard will prevent future unauthorized effects. The detailed reporting correction is in [ERRATA-2026-10-23](ERRATA-2026-10-23.md).
