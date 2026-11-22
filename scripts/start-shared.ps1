param([Parameter(Mandatory=$true)][uri]$ApiUrl)
$ErrorActionPreference = 'Stop'
if ($ApiUrl.Scheme -ne 'https' -or $ApiUrl.UserInfo -or $ApiUrl.Query -or $ApiUrl.Fragment) {
    throw 'Use the shared HTTPS API URL without credentials, query or fragment.'
}
Set-Location (Split-Path -Parent $PSScriptRoot)
$previousApi = $env:VITE_TEAM_API_BASE
$previousE2e = $env:VITE_E2E_MODE
try {
    $env:VITE_TEAM_API_BASE = $ApiUrl.AbsoluteUri.TrimEnd('/')
    $env:VITE_E2E_MODE = 'false'
    if (!(Test-Path 'node_modules/.package-lock.json')) {
        & npm.cmd ci
        if ($LASTEXITCODE -ne 0) { throw 'npm ci failed' }
    }
    & npm.cmd run dev:mobile -- --port 5173 --strictPort
    if ($LASTEXITCODE -ne 0) { throw 'Frontend startup failed' }
} finally {
    $env:VITE_TEAM_API_BASE = $previousApi
    $env:VITE_E2E_MODE = $previousE2e
}
