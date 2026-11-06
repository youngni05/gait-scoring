param([ValidateSet('primary','shift1')][string]$BaseRun)
$ErrorActionPreference='Stop'
$python='C:/Users/jun/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$root='D:/Projects/D_P'
$bundle=Join-Path $root 'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle'
$data=Join-Path 'D:/gait-research-data/D_P/datasets/GAVD/runs' ('confidence_'+$BaseRun)
$small=Join-Path $PSScriptRoot ('runs/confidence_'+$BaseRun)
if ((Test-Path -LiteralPath $data) -or (Test-Path -LiteralPath $small)) { throw 'Refusing to overwrite existing exception run.' }
$old=$env:PYTHONPATH
try {
    $env:PYTHONPATH="$root/tmp/ljmu-cnn-python-deps;$root/tmp/ljmu-analysis-python-deps"
    & $python -B (Join-Path $PSScriptRoot 'prepare_official_confidence.py') --approved --manifest (Join-Path $PSScriptRoot "runs/$BaseRun/approved/records.json") --out (Join-Path $data 'prepared')
    if ($LASTEXITCODE -ne 0) { throw 'Exception adapter preparation failed.' }
    & $python -B (Join-Path $bundle 'infer.py') --input (Join-Path $data 'prepared/prepared.npz') --out (Join-Path $data 'predictions')
    if ($LASTEXITCODE -ne 0) { throw 'Frozen inference failed.' }
    & $python -B (Join-Path $PSScriptRoot 'compare_scores.py') --predictions (Join-Path $data 'predictions/record_predictions.csv') --labels (Join-Path $PSScriptRoot "runs/$BaseRun/approved/labels.csv") --out (Join-Path $small 'comparison')
    if ($LASTEXITCODE -ne 0) { throw 'Comparison failed.' }
} finally { $env:PYTHONPATH=$old }
