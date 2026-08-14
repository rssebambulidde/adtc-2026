# Runs the ADTC profiler on Windows with the local llama.cpp binaries on PATH.
# Usage: pwsh scripts/run_profiler.ps1 [-SkipAccuracy] [-Output submission.json]
param(
    [switch]$SkipAccuracy,
    [string]$Output = "submission.json"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# The profiler prints U+2192/U+2713; the default Windows console codepage cannot
# encode them and rich raises UnicodeEncodeError mid-run.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$llamaDir = Join-Path $root "tools\llama"
if (-not (Test-Path (Join-Path $llamaDir "llama-bench.exe"))) {
    throw "llama-bench.exe not found in $llamaDir. Download the llama.cpp Windows CPU build first."
}
$env:PATH = "$llamaDir;$env:PATH"

$profiler = Join-Path $root ".venv\Scripts\adtc-profiler.exe"
if (-not (Test-Path $profiler)) {
    throw "adtc-profiler not installed in .venv. See README runbook."
}

$modelPath = (Get-Content metadata.json -Raw | ConvertFrom-Json)._runtime.model_path
if (-not (Test-Path $modelPath)) {
    throw "Model not found at $modelPath. Run download_model.sh (or place the GGUF there) first."
}

$args = @("run", "--submission", ".", "--mode", "participant", "--output", $Output)
if ($SkipAccuracy) { $args += "--skip-accuracy" }

Write-Host "Running profiler against $modelPath ..." -ForegroundColor Cyan
& $profiler @args
if ($LASTEXITCODE -ne 0) { throw "profiler exited $LASTEXITCODE" }

& (Join-Path $root ".venv\Scripts\python.exe") scripts/score.py $Output
