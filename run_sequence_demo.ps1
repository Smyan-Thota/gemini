param(
    [Parameter(Mandatory = $true)][string]$QwenBaseModel,
    [string]$InputFile = "examples/sequence-host-demo.jsonl",
    [string]$Out = "runs/sequence-host-demo.jsonl",
    [ValidateSet("cuda", "cpu")][string]$QwenDevice = "cuda",
    [string]$Python = "python"
)
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path $PSScriptRoot "src"
& $Python -m decision_factory.runtime_guard.sequence_cli `
    --input (Join-Path $PSScriptRoot $InputFile) `
    --out (Join-Path $PSScriptRoot $Out) `
    --qwen-base $QwenBaseModel `
    --qwen-device $QwenDevice `
    --bev-device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
