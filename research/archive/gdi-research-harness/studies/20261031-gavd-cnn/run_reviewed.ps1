param([Parameter(Mandatory=$true)][string]$ReviewCsv,
      [Parameter(Mandatory=$true)][ValidatePattern('^[a-zA-Z0-9_-]+$')][string]$RunId)
$ErrorActionPreference='Stop'
$study=$PSScriptRoot
$root=Split-Path (Split-Path (Split-Path $study -Parent) -Parent) -Parent
$python='C:/Users/jun/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$bundle=Join-Path $root 'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle'
$data=Join-Path 'D:/gait-research-data/D_P/datasets/GAVD/runs' $RunId
$small=Join-Path $study ('runs/'+$RunId)
if ((Test-Path -LiteralPath $data) -or (Test-Path -LiteralPath $small)) { throw 'Run ID already exists; do not overwrite.' }
$previousPythonPath=$env:PYTHONPATH
try {
    $env:PYTHONPATH=(Join-Path $root 'tmp/ljmu-cnn-python-deps')+';'+(Join-Path $root 'tmp/ljmu-analysis-python-deps')
    # Reject unreviewed/missing real inputs before starting expensive computation.
    & $python -B (Join-Path $study 'prepare_comparison.py') export --review $ReviewCsv --out (Join-Path $small 'approved')
    if ($LASTEXITCODE -ne 0) { throw 'Reviewed-input export failed.' }
    New-Item -ItemType Directory -Path $data | Out-Null
    & $python -B (Join-Path $bundle 'prepare.py') --manifest (Join-Path $small 'approved/records.json') --out (Join-Path $data 'prepared')
    if ($LASTEXITCODE -ne 0) { throw 'Frozen preparation failed; inspect excluded.csv.' }
    & $python -B (Join-Path $bundle 'infer.py') --input (Join-Path $data 'prepared/prepared.npz') --out (Join-Path $data 'predictions')
    if ($LASTEXITCODE -ne 0) { throw 'Frozen inference failed.' }
    & $python -B (Join-Path $study 'compare_scores.py') --predictions (Join-Path $data 'predictions/record_predictions.csv') --labels (Join-Path $small 'approved/labels.csv') --out (Join-Path $small 'comparison')
    if ($LASTEXITCODE -ne 0) { throw 'Comparison failed.' }
} finally { $env:PYTHONPATH=$previousPythonPath }
