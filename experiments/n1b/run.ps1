param([switch]$TestsOnly)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location -LiteralPath $taskRoot
$env:PATH = "$taskRoot\.conda\nerficg\Library\bin;$env:PATH"
& "$taskRoot\.conda\nerficg\python.exe" -m unittest experiments.n1b.test_core experiments.n1b.test_anchored -v
if ($LASTEXITCODE -ne 0) { throw 'N1b correctness checks failed' }
if (!$TestsOnly) {
    & "$taskRoot\.conda\nerficg\python.exe" -u -m experiments.n1b.benchmark
    if ($LASTEXITCODE -ne 0) { throw 'N1b diagnostic failed' }
}
