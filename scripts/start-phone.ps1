param([Parameter(Mandatory=$true)][System.Net.IPAddress]$Address)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
if ($Address.AddressFamily -ne [System.Net.Sockets.AddressFamily]::InterNetwork) { throw 'Use the Wi-Fi IPv4 address.' }
$phoneHost = $Address.ToString()
if (!(Get-NetIPAddress -AddressFamily IPv4 | Where-Object IPAddress -eq $phoneHost)) { throw 'Address is not assigned to this PC.' }
& "$PSScriptRoot/start-dev.ps1" -BackendOnly
. "$PSScriptRoot/local-api.ps1"
. "$PSScriptRoot/build-local-web.ps1"
$apiUrl = Get-LocalApiUrl
$openssl = 'C:\Program Files\Git\usr\bin\openssl.exe'
if (!(Test-Path -LiteralPath $openssl)) { $openssl = (Get-Command openssl -ErrorAction Stop).Source }
function Invoke-OpenSSL([string[]]$Arguments) {
    & $openssl @Arguments
    if ($LASTEXITCODE -ne 0) { throw 'Certificate operation failed.' }
}
$tlsDir = Join-Path $projectRoot 'local-test-data/phone-tls'
New-Item -ItemType Directory -Path $tlsDir -Force | Out-Null
Push-Location $tlsDir
try {
    if (!(Test-Path 'ca.key') -and !(Test-Path 'ca.crt')) {
        Invoke-OpenSSL @('req','-x509','-newkey','rsa:2048','-nodes','-sha256','-days','365','-keyout','ca.key','-out','ca.crt','-subj','/CN=Happy Aging Local Development CA','-addext','basicConstraints=critical,CA:TRUE,pathlen:0','-addext','keyUsage=critical,keyCertSign,cRLSign')
    }
    if (!(Test-Path 'ca.key') -or !(Test-Path 'ca.crt')) { throw 'Incomplete CA files; restore the existing CA instead of replacing it.' }
    Invoke-OpenSSL @('x509','-checkend','86400','-noout','-in','ca.crt')
    @"
subjectAltName=IP:$phoneHost
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
"@ | Set-Content 'server.ext' -Encoding ascii
    Invoke-OpenSSL @('req','-new','-newkey','rsa:2048','-nodes','-sha256','-keyout','server.key','-out','server.csr','-subj',"/CN=$phoneHost")
    Invoke-OpenSSL @('x509','-req','-in','server.csr','-CA','ca.crt','-CAkey','ca.key','-CAcreateserial','-out','server.crt','-days','90','-sha256','-extfile','server.ext')
    Invoke-OpenSSL @('x509','-in','ca.crt','-noout','-fingerprint','-sha256')
} finally { Pop-Location }
$previousHost = $env:PHONE_HOST
$previousUpstream = $env:PHONE_API_URL
try {
    Invoke-LocalWebBuild -Phone
    $env:PHONE_HOST = $phoneHost
    $env:PHONE_API_URL = $apiUrl
    node scripts/serve-phone.mjs
    if ($LASTEXITCODE -ne 0) { throw 'Phone server stopped with an error' }
} finally {
    $env:PHONE_HOST = $previousHost
    $env:PHONE_API_URL = $previousUpstream
}
