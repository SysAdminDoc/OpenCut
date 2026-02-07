; ============================================================
; OpenCut Installer - Inno Setup Script
;
; Creates a polished Windows installer that:
;   1. Installs the backend server exe (PyInstaller bundle)
;   2. Installs the CEP extension for Premiere Pro
;   3. Sets PlayerDebugMode registry keys for unsigned CEP
;   4. Creates Start Menu + Desktop shortcuts
;   5. Auto-kills any running server during install/uninstall
;   6. Checks for FFmpeg and provides guidance
;   7. Optionally launches the server after install
;
; Build:  iscc build\installer.iss
; Needs:  Inno Setup 6.2+ (https://jrsoftware.org/isinfo.php)
; ============================================================

#define AppName      "OpenCut"
#define AppVersion   "1.0.0"
#define AppPublisher "OpenCut"
#define AppURL       "https://github.com/SysAdminDoc/opencut"
#define AppExeName   "opencut-server.exe"

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
DefaultDirName={localappdata}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir=..\dist
OutputBaseFilename=OpenCut-Setup-{#AppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0

; ---- Visual ----
WizardStyle=modern
WizardSizePercent=110
WizardResizable=no
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\{#AppExeName}

; ---- Branding colors ----
; The modern wizard uses the system accent, but we set what we can
DisableWelcomePage=no
DisableDirPage=no
DisableReadyPage=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Messages]
WelcomeLabel1=Welcome to {#AppName}
WelcomeLabel2=This will install {#AppName} {#AppVersion} on your computer.%n%n{#AppName} is a Premiere Pro plugin that automatically removes silences, filler words, and generates styled captions for your videos.%n%nClick Next to continue.
FinishedHeadingLabel=Installation Complete
FinishedLabelNoIcons={#AppName} has been installed successfully.%n%nOpen Premiere Pro and look for the {#AppName} panel under Window > Extensions.
FinishedLabel={#AppName} has been installed successfully.%n%nOpen Premiere Pro and look for the {#AppName} panel under Window > Extensions.

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"; Flags: unchecked
Name: "startserver"; Description: "Start the {#AppName} server now"; GroupDescription: "After Install:"

[Files]
; Backend server (PyInstaller output directory)
Source: "..\dist\opencut-server\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; CEP Extension
Source: "..\extension\com.opencut.panel\*"; DestDir: "{userappdata}\Adobe\CEP\extensions\com.opencut.panel"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Comment: "Start the {#AppName} backend server"; IconFilename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon; Comment: "Start the {#AppName} backend server"

[Registry]
; Enable unsigned CEP extensions (PlayerDebugMode) for CC 2015 through 2025+
Root: HKCU; Subkey: "Software\Adobe\CSXS.6";  ValueType: string; ValueName: "PlayerDebugMode"; ValueData: "1"; Flags: createvalueifdoesntexist
Root: HKCU; Subkey: "Software\Adobe\CSXS.7";  ValueType: string; ValueName: "PlayerDebugMode"; ValueData: "1"; Flags: createvalueifdoesntexist
Root: HKCU; Subkey: "Software\Adobe\CSXS.8";  ValueType: string; ValueName: "PlayerDebugMode"; ValueData: "1"; Flags: createvalueifdoesntexist
Root: HKCU; Subkey: "Software\Adobe\CSXS.9";  ValueType: string; ValueName: "PlayerDebugMode"; ValueData: "1"; Flags: createvalueifdoesntexist
Root: HKCU; Subkey: "Software\Adobe\CSXS.10"; ValueType: string; ValueName: "PlayerDebugMode"; ValueData: "1"; Flags: createvalueifdoesntexist
Root: HKCU; Subkey: "Software\Adobe\CSXS.11"; ValueType: string; ValueName: "PlayerDebugMode"; ValueData: "1"; Flags: createvalueifdoesntexist
Root: HKCU; Subkey: "Software\Adobe\CSXS.12"; ValueType: string; ValueName: "PlayerDebugMode"; ValueData: "1"; Flags: createvalueifdoesntexist

; Store install path for the panel to locate the exe
Root: HKCU; Subkey: "Software\OpenCut"; ValueType: string; ValueName: "InstallPath"; ValueData: "{app}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\OpenCut"; ValueType: string; ValueName: "Version"; ValueData: "{#AppVersion}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\OpenCut"; ValueType: string; ValueName: "ExePath"; ValueData: "{app}\{#AppExeName}"; Flags: uninsdeletekey

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName} server"; Flags: nowait postinstall; Tasks: startserver

[UninstallDelete]
Type: filesandordirs; Name: "{userappdata}\Adobe\CEP\extensions\com.opencut.panel"
Type: files; Name: "{userprofile}\.opencut\server.pid"
Type: files; Name: "{userprofile}\.opencut\server.log"

[UninstallRun]
Filename: "taskkill"; Parameters: "/F /IM {#AppExeName}"; Flags: runhidden; RunOnceId: "KillServer"

[Code]
// ----------------------------------------------------------------
// Kill any running OpenCut server before install/upgrade/uninstall
// ----------------------------------------------------------------
procedure KillExistingServer;
var
  ResultCode: Integer;
  PidFile: String;
  Pid: String;
  Lines: TArrayOfString;
begin
  // Method 1: Kill by exe name
  Exec('taskkill', '/F /T /IM ' + '{#AppExeName}', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);

  // Method 2: Kill by PID file
  PidFile := ExpandConstant('{userprofile}\.opencut\server.pid');
  if FileExists(PidFile) then
  begin
    if LoadStringsFromFile(PidFile, Lines) and (GetArrayLength(Lines) > 0) then
    begin
      Pid := Trim(Lines[0]);
      if Pid <> '' then
        Exec('taskkill', '/F /T /PID ' + Pid, '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
    end;
    DeleteFile(PidFile);
  end;

  // Method 3: Kill any python running our module
  Exec('taskkill', '/F /FI "WINDOWTITLE eq OpenCut*"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);

  Sleep(800);
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  KillExistingServer;
  Result := '';
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then
    KillExistingServer;
end;

// ----------------------------------------------------------------
// Post-install: check for FFmpeg
// ----------------------------------------------------------------
function FFmpegInPath: Boolean;
var
  ResultCode: Integer;
begin
  Result := Exec('cmd', '/C ffmpeg -version', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    if not FFmpegInPath then
    begin
      MsgBox(
        'FFmpeg was not found on your system.' + #13#10 + #13#10 +
        'OpenCut requires FFmpeg for audio/video processing.' + #13#10 + #13#10 +
        'Install it with one of these methods:' + #13#10 +
        '  - Open a terminal and run: winget install ffmpeg' + #13#10 +
        '  - Download from: https://ffmpeg.org/download.html' + #13#10 + #13#10 +
        'After installing FFmpeg, restart your computer.',
        mbInformation, MB_OK
      );
    end;
  end;
end;
