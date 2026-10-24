param([Parameter(Mandatory=$true)][string]$QwenBaseModel,[string]$InputFile="examples/sequence-host-demo.jsonl",[string]$Out="runs/sequence-host-demo.jsonl",[ValidateSet("cuda","cpu")][string]$QwenDevice="cuda",[ValidateSet("cuda")][string]$BevDevice="cuda",[string]$Python="python")
$ErrorActionPreference="Stop";$env:PYTHONPATH=Join-Path $PSScriptRoot "src"
& $Python -m decision_factory.runtime_guard.sequence_host_demo --input (Join-Path $PSScriptRoot $InputFile) --out (Join-Path $PSScriptRoot $Out) --qwen-base $QwenBaseModel --qwen-device $QwenDevice --bev-device $BevDevice
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
