function Invoke-LocalWebBuild {
    param([switch]$Phone)
    $savedApi = $env:VITE_TEAM_API_BASE
    $savedE2e = $env:VITE_E2E_MODE
    try {
        # Both web servers proxy /api on their own origin. Never bake in a LAN IP.
        $env:VITE_TEAM_API_BASE = '/api'
        $env:VITE_E2E_MODE = 'false'
        $buildArgs = @('run', 'build:mobile')
        if ($Phone) { $buildArgs += @('--', '--outDir', 'local-output/phone') }
        & npm.cmd @buildArgs
        if ($LASTEXITCODE -ne 0) { throw 'Local web build failed.' }
    } finally {
        $env:VITE_TEAM_API_BASE = $savedApi
        $env:VITE_E2E_MODE = $savedE2e
    }
}
