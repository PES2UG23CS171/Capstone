; Inno Setup script for the Windows installer (see build_windows.ps1).
; Installs per-user, so it needs no administrator rights.

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif

[Setup]
AppId={{6F4B2C1E-8D3A-4E57-9A41-1C90D7E2B190}
AppName=VoiceISO
AppVersion={#AppVersion}
AppPublisher=Team 190, PES University
DefaultDirName={localappdata}\Programs\VoiceISO
DefaultGroupName=VoiceISO
DisableProgramGroupPage=yes
DisableDirPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=VoiceISO-{#AppVersion}-Windows-x64-Setup
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\VoiceISO.exe
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
InfoAfterFile=READ_ME_FIRST.txt

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"

[Files]
Source: "..\build\dist\VoiceISO\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "READ_ME_FIRST.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\VoiceISO"; Filename: "{app}\VoiceISO.exe"
Name: "{autodesktop}\VoiceISO"; Filename: "{app}\VoiceISO.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\VoiceISO.exe"; Description: "Open VoiceISO"; Flags: nowait postinstall skipifsilent
