# Run once from an Administrator PowerShell; only the two local preview ports are allowed.
param([Parameter(Mandatory=$true)][System.Net.IPAddress]$Address)
$ErrorActionPreference = 'Stop'
$phoneHost = $Address.ToString()
if ($Address.AddressFamily -ne [System.Net.Sockets.AddressFamily]::InterNetwork -or
    !(Get-NetIPAddress -AddressFamily IPv4 | Where-Object IPAddress -eq $phoneHost)) {
    throw 'Specify this PC Wi-Fi IPv4 address.'
}
$ruleName = 'Happy Aging phone preview'
$existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
if ($existing) {
    $existing | Get-NetFirewallAddressFilter | Set-NetFirewallAddressFilter -LocalAddress $phoneHost -RemoteAddress LocalSubnet
    $existing | Enable-NetFirewallRule
} else {
    New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow -Protocol TCP `
        -LocalPort 5443,5080 -LocalAddress $phoneHost -RemoteAddress LocalSubnet -Profile Any | Out-Null
}
Write-Host "Local subnet access enabled for ${phoneHost}:5443 and :5080."
