param([switch]$BackendOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed (exit $LASTEXITCODE)." }
}

if (!(Test-Path -LiteralPath '.env')) {
    throw 'Missing .env. Follow backend/README.md once; do not overwrite an existing .env.'
}
Invoke-Checked 'docker' @('info', '--format', '{{.ServerVersion}}')
$composeArgs = @('compose', '-f', 'compose.yaml', '-f', 'compose.dev.yaml')
Invoke-Checked 'docker' ($composeArgs + @('up', '-d', '--build', '--wait'))
Invoke-Checked 'docker' ($composeArgs + @('exec', '-T', 'api', 'alembic', 'upgrade', 'head'))
$health = Invoke-RestMethod 'http://127.0.0.1:8000/db-health'
if ($health.database -ne 'connected') { throw 'API database health check failed.' }
Write-Host 'Persistent API ready: http://127.0.0.1:8000'
if ($BackendOnly) { return }

if (!(Test-Path -LiteralPath 'node_modules/.package-lock.json')) {
    Invoke-Checked 'npm.cmd' @('ci')
}
$previousApi = $env:VITE_API_BASE_URL
$previousE2e = $env:VITE_E2E_MODE
try {
    $env:VITE_API_BASE_URL = 'http://127.0.0.1:8000'
    $env:VITE_E2E_MODE = 'false'
    Write-Host 'Open http://127.0.0.1:5173 and sign in with your existing account.'
    Write-Host 'Ctrl+C stops the frontend. To stop API/DB: docker compose stop'
    Invoke-Checked 'npm.cmd' @('run', 'dev:mobile', '--', '--port', '5173', '--strictPort')
} finally {
    $env:VITE_API_BASE_URL = $previousApi
    $env:VITE_E2E_MODE = $previousE2e
}
