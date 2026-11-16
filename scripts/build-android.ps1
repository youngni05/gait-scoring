param(
    [string]$SdkRoot = (Join-Path $env:USERPROFILE 'AppData/Local/Android/Sdk'),
    [string]$JdkRoot = 'C:/Program Files/Java/jdk-21'
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath "$SdkRoot/platform-tools/adb.exe")) { throw "Android SDK not found at $SdkRoot" }
if (-not (Test-Path -LiteralPath "$JdkRoot/bin/java.exe")) { throw "JDK 21 not found at $JdkRoot" }
$previousJava = $env:JAVA_HOME
$previousAndroid = $env:ANDROID_HOME
Push-Location (Join-Path $PSScriptRoot '..')
try {
    $env:JAVA_HOME = $JdkRoot
    $env:ANDROID_HOME = $SdkRoot
    & npm.cmd run android:sync
    if ($LASTEXITCODE -ne 0) { throw 'Web build / Capacitor sync failed' }
    Push-Location mobile/android
    try {
        & .\gradlew.bat assembleDebug --no-daemon
        if ($LASTEXITCODE -ne 0) { throw 'Android build failed' }
    } finally { Pop-Location }
} finally {
    Pop-Location
    $env:JAVA_HOME = $previousJava
    $env:ANDROID_HOME = $previousAndroid
}
