param([switch]$BackendOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
. "$PSScriptRoot/local-api.ps1"

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed (exit $LASTEXITCODE)." }
}

if (!(Test-Path -LiteralPath '.env')) {
    throw 'Missing .env. Follow backend/README.md once; do not overwrite an existing .env.'
}
if (!(Test-Path -LiteralPath 'node_modules/.package-lock.json')) {
    Invoke-Checked 'npm.cmd' @('ci')
}
. "$PSScriptRoot/build-local-web.ps1"
Invoke-LocalWebBuild
Invoke-Checked 'docker' @('info', '--format', '{{.ServerVersion}}')
$composeArgs = @('compose', '-f', 'compose.yaml', '-f', 'compose.dev.yaml')
Invoke-Checked 'docker' ($composeArgs + @('up', '-d', '--build', '--wait'))
# The Compose migrate service applies migrations before API/worker startup.
$apiUrl = Get-LocalApiUrl
Wait-LocalApi -ApiUrl $apiUrl
Write-Host "Persistent API ready: $apiUrl"
if ($BackendOnly) { return }

$previousApi = $env:VITE_TEAM_API_BASE
$previousE2e = $env:VITE_E2E_MODE
try {
    $env:VITE_TEAM_API_BASE = $apiUrl
    $env:VITE_E2E_MODE = 'false'
    Write-Host 'Open http://127.0.0.1:5173 and sign in with your existing account.'
    Write-Host 'Ctrl+C stops the frontend. To stop API/DB: docker compose stop'
    Invoke-Checked 'npm.cmd' @('run', 'dev:mobile', '--', '--port', '5173', '--strictPort')
} finally {
    $env:VITE_TEAM_API_BASE = $previousApi
    $env:VITE_E2E_MODE = $previousE2e
}
