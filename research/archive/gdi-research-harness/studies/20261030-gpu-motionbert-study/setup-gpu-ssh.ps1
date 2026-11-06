# Run interactively as the same Windows user, in Administrator PowerShell.
# Secrets are entered only into OpenSSH prompts; this script never reads them.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Open PowerShell as Administrator under your usual Windows account, then run this script again.'
}
if ($env:USERPROFILE.TrimEnd('\') -ine 'C:\Users\jun') {
    throw 'Run this script as the Windows user whose profile is C:\Users\jun.'
}

$sshDir = Join-Path $env:USERPROFILE '.ssh'
$keyPath = Join-Path $sshDir 'id_ed25519_gdi_abrm02'
$publicPath = "$keyPath.pub"
$sshExe = Join-Path $env:WINDIR 'System32\OpenSSH\ssh.exe'
$keygenExe = Join-Path $env:WINDIR 'System32\OpenSSH\ssh-keygen.exe'
$addExe = Join-Path $env:WINDIR 'System32\OpenSSH\ssh-add.exe'
foreach ($program in @($sshExe, $keygenExe, $addExe)) {
    if (-not (Test-Path -LiteralPath $program)) { throw "Missing OpenSSH tool: $program" }
}

Write-Host '[1/5] Starting Windows SSH authentication agent.'
Set-Service -Name ssh-agent -StartupType Automatic
Start-Service -Name ssh-agent

Write-Host '[2/5] Preparing a dedicated GPU research key.'
New-Item -ItemType Directory -Force -Path $sshDir | Out-Null
if ((Test-Path -LiteralPath $keyPath) -ne (Test-Path -LiteralPath $publicPath)) {
    throw 'Only one file of the dedicated key pair exists. No file was overwritten. Ask for a check.'
}
if (-not (Test-Path -LiteralPath $keyPath)) {
    Write-Host 'Enter a NEW non-empty key passphrase twice. This is not the server password.'
    & $keygenExe -t ed25519 -a 100 -f $keyPath -C 'gdi-RESEARCH_USER-ABRM02'
    if ($LASTEXITCODE -ne 0) { throw 'Key generation failed.' }
} else {
    Write-Host 'Dedicated key already exists; keeping it.'
}

Write-Host '[3/5] Loading the key into ssh-agent. Enter the key passphrase if requested.'
& $addExe $keyPath
if ($LASTEXITCODE -ne 0) { throw 'Could not load the dedicated key into ssh-agent.' }

# Read only the PUBLIC key. Rebuild its comment so remote shell quoting is fixed.
$publicText = (Get-Content -LiteralPath $publicPath -Raw).Trim()
if ($publicText -notmatch '^ssh-ed25519 ([A-Za-z0-9+/]+={0,2})(?:\s+[^\r\n]*)?$') {
    throw 'Unexpected public key format.'
}
$publicLine = 'ssh-ed25519 ' + $Matches[1] + ' gdi-RESEARCH_USER-ABRM02'
$remoteScript = @'
set -eu
umask 077
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
touch "$HOME/.ssh/authorized_keys"
chmod 600 "$HOME/.ssh/authorized_keys"
key='__PUBLIC_KEY__'
if ! grep -qxF "$key" "$HOME/.ssh/authorized_keys"; then
    printf '\n%s\n' "$key" >> "$HOME/.ssh/authorized_keys"
fi
printf 'GPU_PUBLIC_KEY_REGISTERED\n'
'@
$remoteScript = $remoteScript.Replace('__PUBLIC_KEY__', $publicLine).Replace("`r`n", "`n")
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remoteScript))
$connection = @('-p', '10000', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15', '-o', 'ForwardAgent=no')

Write-Host '[4/5] Registering the PUBLIC key. Enter the EXISTING server password when prompted.'
# Require the already trusted host key; never bypass a changed or unknown host key.
& $sshExe @connection -o PubkeyAuthentication=no -o PreferredAuthentications=keyboard-interactive,password RESEARCH_USER@PRIVATE_HOST "echo $encoded | base64 -d | sh"
if ($LASTEXITCODE -ne 0) { throw 'Public key registration failed. Keep the error text, but do not share passwords.' }

Write-Host '[5/5] Checking unattended public-key authentication (no password fallback).'
# A public identity file selects the matching private key held by ssh-agent.
& $sshExe @connection -o BatchMode=yes -o IdentitiesOnly=yes -i $publicPath RESEARCH_USER@PRIVATE_HOST 'hostname; whoami; printf GPU_SSH_KEY_READY'
if ($LASTEXITCODE -ne 0) { throw 'Key was registered, but unattended authentication did not pass.' }
Write-Host "`nSetup passed. Return to the chat and report GPU_SSH_KEY_READY."
