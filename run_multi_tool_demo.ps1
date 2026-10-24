param(
    [Parameter(Mandatory=$true)][string]$InputFile,
    [Parameter(Mandatory=$true)][string]$QwenBaseModel,
    [string]$QwenAdapter,
    [Parameter(Mandatory=$true)][string]$ReadoutCheckpoint,
    [Parameter(Mandatory=$true)][string]$ReadoutSha256,
    [string]$Out="runs/multi-tool-demo.jsonl",
    [ValidateSet("qwen-bev","qwen-alone","matched")][string]$Mode="qwen-bev",
    [ValidateSet("frozen","adapted")][string]$ReadoutKind="frozen",
    [ValidateSet("finite_mean_logprob","generate")][string]$QwenProposalMode="finite_mean_logprob",
    [ValidateSet("legacy","compact","native")][string]$ActorPromptProfile="legacy",
    [switch]$DisableQwenAdapter,
    [switch]$FactualProgress,
    [ValidateSet("cuda","cpu")][string]$QwenDevice="cuda",
    [ValidateSet("cuda","cpu")][string]$BevDevice="cuda",
    [string]$Python="python"
)
$ErrorActionPreference="Stop"
$env:PYTHONPATH=Join-Path $PSScriptRoot "src"
$extra=@()
if ($DisableQwenAdapter) { $extra += "--disable-qwen-adapter" }
if ($FactualProgress) { $extra += "--factual-progress" }
if ($QwenAdapter) { $extra += @("--qwen-adapter", $QwenAdapter) }
& $Python -m decision_factory.runtime_guard.multi_tool_cli `
    --input $InputFile --out $Out --mode $Mode --qwen-base $QwenBaseModel `
    --qwen-device $QwenDevice --qwen-proposal-mode $QwenProposalMode `
    --actor-prompt-profile $ActorPromptProfile `
    --readout-checkpoint $ReadoutCheckpoint --readout-sha256 $ReadoutSha256 `
    --readout-kind $ReadoutKind --bev-device $BevDevice @extra
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
