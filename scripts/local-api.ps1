# Read the actual published port, including Compose .env and shell overrides.
function Get-LocalApiUrl {
    $binding = & docker compose -f compose.yaml -f compose.dev.yaml port api 8000
    if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve the running API port.' }
    $binding = ($binding | Out-String).Trim()
    if ($binding -notmatch '^127\.0\.0\.1:(\d+)$') {
        throw 'Expected one API port bound to 127.0.0.1.'
    }
    return "http://$binding"
}

# Container running does not necessarily mean Uvicorn and the DB are ready.
function Wait-LocalApi {
    param(
        [Parameter(Mandatory=$true)][string]$ApiUrl,
        [ValidateRange(1, 300)][int]$TimeoutSeconds = 60
    )
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    $lastStatus = 'No health response received.'
    Write-Host "Waiting for API and database readiness (up to ${TimeoutSeconds}s)..."
    try {
        while ($timer.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
            $remaining = $TimeoutSeconds - $timer.Elapsed.TotalSeconds
            if ($remaining -lt 1) { break }
            $requestTimeout = [int][Math]::Min(5, [Math]::Floor($remaining))
            try {
                $health = Invoke-RestMethod -Uri "$ApiUrl/db-health" -TimeoutSec $requestTimeout -ErrorAction Stop
                if ($health.database -eq 'connected') { return }
                $lastStatus = 'API responded, but database readiness was not confirmed.'
            } catch {
                # Do not print response bodies, which may contain internal details.
                $lastStatus = 'Health request failed or timed out.'
            }
            $delay = [int][Math]::Min(1000, [Math]::Max(0, ($TimeoutSeconds - $timer.Elapsed.TotalSeconds) * 1000))
            if ($delay -gt 0) { Start-Sleep -Milliseconds $delay }
        }
        throw "API readiness timeout at $ApiUrl after ${TimeoutSeconds}s. $lastStatus Check: docker compose -f compose.yaml -f compose.dev.yaml logs --tail 80 api db migrate"
    } finally {
        $timer.Stop()
    }
}
