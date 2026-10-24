# Matched Stage C candidate run

`tools/run_stage_c_matched_candidate.py` runs the frozen 28-task Stage C suite in three arms: H17 with the structural host only, H17 with the frozen V7 guard, and H17 with an operator-pinned candidate Bev checkpoint. All arms use the same H17 adapter, base snapshot, production prompt renderer, `finite_mean_logprob` mode, and canonical ACT/REVIEW/STOP scorer. Proposals are replayed across arms only when the exact rendered prompt SHA-256 matches; after histories diverge, each arm gets its own model-scored proposal. The scorer is a deterministic menu-selection heuristic, not a calibrated confidence measure.

The input is the frozen local cohort at `%LOCALAPPDATA%\gemini\stage-c-v2\episodes.jsonl` (SHA-256 `198cf0560ab8dce1d09d5a88d92c42aff5378687c03c373f052068c09d2065f2`) and its manifest. The script verifies both before loading models. Outputs are immutable per-arm JSONL traces and a receipt in a new empty directory. Effects are synthetic in-memory state only.

Use the existing local inference environment and cached base snapshots. Substitute the selected candidate file and exact SHA from its selection receipt; do not use a checkpoint that has not passed the current single-request assessor gates.

Preflight the exact local files without loading models:

```powershell
$repo = 'C:\Users\smyan\OneDrive\Desktop\gemini'
$python = 'C:\Users\smyan\OneDrive\Desktop\bevthing\.venv\Scripts\python.exe'
$env:PYTHONPATH = Join-Path $repo 'src'
$candidate = 'C:\path\to\selected-candidate.pt'
$candidateSha = 'replace-with-the-exact-64-character-checkpoint-sha256'
$preflightOut = Join-Path $env:LOCALAPPDATA 'gemini\stage-c-v3\matched-preflight'

& $python (Join-Path $repo 'tools\run_stage_c_matched_candidate.py') `
  --preflight-only --out-dir $preflightOut `
  --candidate-checkpoint $candidate --candidate-sha256 $candidateSha
if ($LASTEXITCODE -ne 0) { throw "Matched Stage C preflight failed: $LASTEXITCODE" }
```

After the preflight passes and the selected model is frozen, run the inference command:

```powershell
$repo = 'C:\Users\smyan\OneDrive\Desktop\gemini'
$python = 'C:\Users\smyan\OneDrive\Desktop\bevthing\.venv\Scripts\python.exe'
$env:PYTHONPATH = Join-Path $repo 'src'
$candidate = 'C:\path\to\selected-candidate.pt'
$candidateSha = 'replace-with-the-exact-64-character-checkpoint-sha256'
$output = Join-Path $env:LOCALAPPDATA 'gemini\stage-c-v3\matched-candidate-run-01'

& $python (Join-Path $repo 'tools\run_stage_c_matched_candidate.py') `
  --out-dir $output `
  --candidate-checkpoint $candidate `
  --candidate-sha256 $candidateSha
if ($LASTEXITCODE -ne 0) { throw "Matched Stage C run failed: $LASTEXITCODE" }
```

Run `--preflight-only` with the same paths first to check the case manifest, episode count, and checkpoint identities without loading either model. Keep all trace and receipt files outside the public source tree; send them to the independent Stage C scorer for the final tally. A completed command is not a promotion by itself: report each arm's target completion, safe completion, unauthorized synthetic effects, review and error counts, plus prompt-cache statistics.
