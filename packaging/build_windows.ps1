# Build dist\VoiceISO-<version>-Windows-x64-Setup.exe on Windows.  Needs the build
# environment's python on PATH and Inno Setup 6 (choco install innosetup).
#
# Not code-signed: removing the SmartScreen warning needs a code-signing certificate.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Invoke-SelfTest([string]$Exe, [string]$Report) {
    # Windowed exes detach from the console, so wait explicitly for the exit code.
    $p = Start-Process -FilePath $Exe -ArgumentList "--self-test", "--self-test-out", "`"$Report`"" `
        -Wait -PassThru -RedirectStandardOutput "$Report.stdout" -RedirectStandardError "$Report.stderr"
    if (Test-Path $Report) { Get-Content $Report } else { Get-Content "$Report.stdout", "$Report.stderr" }
    # The audio child logs to the user's app-data folder.  Keep it beside the report.
    $Log = "$env:LOCALAPPDATA\VoiceISO\voiceiso.log"
    if (Test-Path $Log) { Copy-Item $Log "$Report.log" }
    if ($p.ExitCode -ne 0) { throw "Self-test failed for $Exe (exit $($p.ExitCode))" }
}

$Version = (Get-Content packaging\VERSION -Raw).Trim()
$Root = (Get-Location).Path
New-Item -ItemType Directory -Force -Path build, dist | Out-Null

python -m PyInstaller --noconfirm --clean packaging\voiceiso.spec --distpath build\dist --workpath build\work
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

Write-Host "--- Microsoft C++ runtime: must ship (clean PCs may lack it), and only once (see voiceiso.spec)"
foreach ($dll in "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll") {
    $Copies = @(Get-ChildItem build\dist\VoiceISO -Recurse -Filter $dll)
    $Copies | ForEach-Object { Write-Host "$($_.FullName)  $($_.VersionInfo.FileVersion)" }
    if ($Copies.Count -ne 1) { throw "$dll must ship exactly once, found $($Copies.Count)" }
}

Write-Host "--- self-test of the built app"
Invoke-SelfTest "$Root\build\dist\VoiceISO\VoiceISO.exe" "$Root\build\selftest-windows.json"

Write-Host "--- installer"
$Iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
& $Iscc "/DAppVersion=$Version" packaging\installer.iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }
$Setup = "$Root\dist\VoiceISO-$Version-Windows-x64-Setup.exe"

Write-Host "--- install it as a user would, then self-test the installed copy"
$p = Start-Process -FilePath $Setup -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART" -Wait -PassThru
if ($p.ExitCode -ne 0) { throw "Installer failed (exit $($p.ExitCode))" }
Invoke-SelfTest "$env:LOCALAPPDATA\Programs\VoiceISO\VoiceISO.exe" "$Root\build\selftest-windows-installed.json"

Get-Item $Setup | Select-Object Name, @{ n = "MB"; e = { [int]($_.Length / 1MB) } }
