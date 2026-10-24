param([Parameter(Mandatory=$true)][string]$QwenBaseModel,[string]$InputFile="examples/sequence-demo.jsonl",[string]$Out="runs/sequence-demo.jsonl",[ValidateSet("cuda","cpu")][string]$QwenDevice="cuda",[ValidateSet("cuda","cpu")][string]$BevDevice="cuda",[string]$Python="python")
$ErrorActionPreference="Stop"
$env:PYTHONPATH=Join-Path $PSScriptRoot "src"
& $Python -m decision_factory.runtime_guard.sequence_cli --input (Join-Path $PSScriptRoot $InputFile) --out (Join-Path $PSScriptRoot $Out) --qwen-base $QwenBaseModel --qwen-device $QwenDevice --bev-device $BevDevice
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
