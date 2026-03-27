; OpenCut Installer Script for Inno Setup 6
; Fully self-contained installer — bundles server exe, ffmpeg, and CEP extension

#define MyAppName "OpenCut"
#define MyAppVersion "1.9.9"
#define MyAppPublisher "SysAdminDoc"
#define MyAppURL "https://github.com/SysAdminDoc/OpenCut"

[Setup]
AppId={{8A7B9C0D-1E2F-3A4B-5C6D-7E8F9A0B1C2D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} v{#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile=LICENSE
OutputDir=installer\dist
OutputBaseFilename=OpenCut-Setup-{#MyAppVersion}
SetupIconFile=img\logo.ico
UninstallDisplayIcon={app}\logo.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesInstallIn64BitMode=x64compatible
WizardImageFile=compiler:WizClassicImage-IS.bmp
WizardSmallImageFile=compiler:WizClassicSmallImage-IS.bmp
ChangesEnvironment=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "autostart"; Description: "Start OpenCut server when Windows starts"; GroupDescription: "Startup:"
Name: "installextension"; Description: "Install Adobe Premiere Pro CEP extension"; GroupDescription: "Adobe Integration:"
Name: "downloadmodel"; Description: "Download Whisper AI model for captions (~1.6GB, requires internet)"; GroupDescription: "AI Models:"
Name: "downloadmodel\tiny"; Description: "tiny (75MB) — Fastest, lower accuracy"; GroupDescription: "AI Models:"; Flags: exclusive unchecked
Name: "downloadmodel\base"; Description: "base (150MB) — Good balance"; GroupDescription: "AI Models:"; Flags: exclusive unchecked
Name: "downloadmodel\small"; Description: "small (500MB) — Better accuracy, slower"; GroupDescription: "AI Models:"; Flags: exclusive unchecked
Name: "downloadmodel\medium"; Description: "medium (1.5GB) — High accuracy, requires more RAM"; GroupDescription: "AI Models:"; Flags: exclusive unchecked
Name: "downloadmodel\turbo"; Description: "turbo (1.6GB) — Best speed/accuracy ratio (recommended)"; GroupDescription: "AI Models:"; Flags: exclusive

[Files]
; Icon
Source: "img\logo.ico"; DestDir: "{app}"; Flags: ignoreversion

; Bundled server (PyInstaller output — includes Python runtime + all deps)
Source: "dist\OpenCut-Server\*"; DestDir: "{app}\server"; Flags: ignoreversion recursesubdirs createallsubdirs

; Bundled FFmpeg (ffmpeg.exe + ffprobe.exe)
Source: "ffmpeg\ffmpeg.exe"; DestDir: "{app}\ffmpeg"; Flags: ignoreversion
Source: "ffmpeg\ffprobe.exe"; DestDir: "{app}\ffmpeg"; Flags: ignoreversion

; Hidden launcher (runs server with no console window)
Source: "OpenCut-Launcher.vbs"; DestDir: "{app}"; Flags: ignoreversion

; CEP Extension
Source: "extension\com.opencut.panel\*"; DestDir: "{app}\extension\com.opencut.panel"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
Name: "{app}\logs"

[Icons]
; Start menu — hidden launcher (no console window)
Name: "{group}\OpenCut Server"; Filename: "wscript.exe"; Parameters: """{app}\OpenCut-Launcher.vbs"""; WorkingDir: "{app}"; IconFilename: "{app}\logo.ico"
Name: "{group}\OpenCut Server (Console)"; Filename: "{app}\server\OpenCut-Server.exe"; WorkingDir: "{app}\server"; IconFilename: "{app}\logo.ico"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
; Desktop shortcut — hidden launcher
Name: "{autodesktop}\OpenCut"; Filename: "wscript.exe"; Parameters: """{app}\OpenCut-Launcher.vbs"""; WorkingDir: "{app}"; IconFilename: "{app}\logo.ico"; Tasks: desktopicon
; Startup — hidden launcher
Name: "{userstartup}\OpenCut"; Filename: "wscript.exe"; Parameters: """{app}\OpenCut-Launcher.vbs"""; WorkingDir: "{app}"; Tasks: autostart

[Registry]
; App install path (for detection by other tools)
Root: HKCU; Subkey: "Software\{#MyAppName}"; ValueType: string; ValueName: "InstallPath"; ValueData: "{app}"; Flags: uninsdeletekey

; Enable unsigned CEP extensions (PlayerDebugMode) for CSXS 7-12
Root: HKCU; Subkey: "Software\Adobe\CSXS.7"; ValueType: string; ValueName: "PlayerDebugMode"; ValueData: "1"; Flags: createvalueifdoesntexist; Tasks: installextension
Root: HKCU; Subkey: "Software\Adobe\CSXS.8"; ValueType: string; ValueName: "PlayerDebugMode"; ValueData: "1"; Flags: createvalueifdoesntexist; Tasks: installextension
Root: HKCU; Subkey: "Software\Adobe\CSXS.9"; ValueType: string; ValueName: "PlayerDebugMode"; ValueData: "1"; Flags: createvalueifdoesntexist; Tasks: installextension
Root: HKCU; Subkey: "Software\Adobe\CSXS.10"; ValueType: string; ValueName: "PlayerDebugMode"; ValueData: "1"; Flags: createvalueifdoesntexist; Tasks: installextension
Root: HKCU; Subkey: "Software\Adobe\CSXS.11"; ValueType: string; ValueName: "PlayerDebugMode"; ValueData: "1"; Flags: createvalueifdoesntexist; Tasks: installextension
Root: HKCU; Subkey: "Software\Adobe\CSXS.12"; ValueType: string; ValueName: "PlayerDebugMode"; ValueData: "1"; Flags: createvalueifdoesntexist; Tasks: installextension

[Run]
; Download Whisper model (runs with visible console so user can see progress)
Filename: "{app}\server\OpenCut-Server.exe"; Parameters: "--download-models tiny"; StatusMsg: "Downloading Whisper AI model (tiny)..."; Flags: runasoriginaluser; Tasks: downloadmodel\tiny
Filename: "{app}\server\OpenCut-Server.exe"; Parameters: "--download-models base"; StatusMsg: "Downloading Whisper AI model (base)..."; Flags: runasoriginaluser; Tasks: downloadmodel\base
Filename: "{app}\server\OpenCut-Server.exe"; Parameters: "--download-models small"; StatusMsg: "Downloading Whisper AI model (small)..."; Flags: runasoriginaluser; Tasks: downloadmodel\small
Filename: "{app}\server\OpenCut-Server.exe"; Parameters: "--download-models medium"; StatusMsg: "Downloading Whisper AI model (medium)..."; Flags: runasoriginaluser; Tasks: downloadmodel\medium
Filename: "{app}\server\OpenCut-Server.exe"; Parameters: "--download-models turbo"; StatusMsg: "Downloading Whisper AI model (turbo)..."; Flags: runasoriginaluser; Tasks: downloadmodel\turbo
; Start server after install
Filename: "wscript.exe"; Parameters: """{app}\OpenCut-Launcher.vbs"""; Description: "Start OpenCut Server"; Flags: nowait postinstall skipifsilent

[Code]
const
  EnvironmentKey = 'Environment';

// --- Kill OpenCut server processes ---

procedure KillOpenCutProcesses();
var
  ResultCode: Integer;
begin
  // Kill all OpenCut-Server.exe processes
  Exec('taskkill.exe', '/F /IM OpenCut-Server.exe', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  // Kill any wscript running our launcher
  Exec('taskkill.exe', '/F /FI "WINDOWTITLE eq OpenCut*"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  // Kill python server if running from source
  Exec('cmd.exe', '/c for /f "tokens=5" %a in (''netstat -ano ^| findstr :5679 ^| findstr LISTENING'') do taskkill /F /PID %a', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  // Brief pause to let processes die
  Sleep(500);
end;

// --- PATH management for bundled FFmpeg ---

procedure AddToPath(Dir: string);
var
  OldPath: string;
begin
  if not RegQueryStringValue(HKCU, EnvironmentKey, 'Path', OldPath) then
    OldPath := '';
  if Pos(Uppercase(Dir), Uppercase(OldPath)) = 0 then
  begin
    if OldPath <> '' then
      OldPath := OldPath + ';';
    RegWriteStringValue(HKCU, EnvironmentKey, 'Path', OldPath + Dir);
  end;
end;

procedure RemoveFromPath(Dir: string);
var
  OldPath, UpperDir, UpperPath: string;
  P: Integer;
begin
  if not RegQueryStringValue(HKCU, EnvironmentKey, 'Path', OldPath) then
    Exit;
  UpperDir := Uppercase(Dir);
  UpperPath := Uppercase(OldPath);
  P := Pos(UpperDir, UpperPath);
  if P > 0 then
  begin
    Delete(OldPath, P, Length(Dir));
    while Pos(';;', OldPath) > 0 do
      StringChangeEx(OldPath, ';;', ';', True);
    if (Length(OldPath) > 0) and (OldPath[1] = ';') then
      Delete(OldPath, 1, 1);
    if (Length(OldPath) > 0) and (OldPath[Length(OldPath)] = ';') then
      Delete(OldPath, Length(OldPath), 1);
    RegWriteStringValue(HKCU, EnvironmentKey, 'Path', OldPath);
  end;
end;

// --- CEP extension copy ---

function DirectoryCopy(SourcePath, DestPath: string): Boolean;
var
  FindRec: TFindRec;
  SourceFilePath, DestFilePath: string;
begin
  Result := True;
  if not DirExists(DestPath) then
    if not CreateDir(DestPath) then
    begin
      Result := False;
      Exit;
    end;
  if FindFirst(SourcePath + '\*', FindRec) then
  begin
    try
      repeat
        if (FindRec.Name <> '.') and (FindRec.Name <> '..') then
        begin
          SourceFilePath := SourcePath + '\' + FindRec.Name;
          DestFilePath := DestPath + '\' + FindRec.Name;
          if FindRec.Attributes and FILE_ATTRIBUTE_DIRECTORY <> 0 then
            Result := DirectoryCopy(SourceFilePath, DestFilePath) and Result
          else
            Result := CopyFile(SourceFilePath, DestFilePath, False) and Result;
        end;
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
  end;
end;

procedure InstallCEPExtension();
var
  ExtSrc, ExtDest, ExtParent: string;
begin
  ExtSrc := ExpandConstant('{app}\extension\com.opencut.panel');
  ExtParent := ExpandConstant('{userappdata}\Adobe\CEP\extensions');
  ExtDest := ExtParent + '\com.opencut.panel';

  if not DirExists(ExtParent) then
    ForceDirectories(ExtParent);

  if DirExists(ExtDest) then
    DelTree(ExtDest, True, True, True);

  DirectoryCopy(ExtSrc, ExtDest);
end;

// --- Pre-install: kill server before upgrading ---

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  KillOpenCutProcesses();
  Result := '';
end;

// --- Post-install hooks ---

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    // Add bundled FFmpeg to user PATH
    AddToPath(ExpandConstant('{app}\ffmpeg'));
    // Install CEP extension
    if WizardIsTaskSelected('installextension') then
      InstallCEPExtension();
  end;
end;

// --- Uninstall hooks ---

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  ExtPath, ConfigDir, StartupShortcut: string;
  ResultCode: Integer;
begin
  if CurUninstallStep = usUninstall then
  begin
    // Kill running server processes first
    KillOpenCutProcesses();

    // Remove CEP extension from Adobe folder
    ExtPath := ExpandConstant('{userappdata}\Adobe\CEP\extensions\com.opencut.panel');
    if DirExists(ExtPath) then
      DelTree(ExtPath, True, True, True);

    // Remove FFmpeg from user PATH
    RemoveFromPath(ExpandConstant('{app}\ffmpeg'));

    // Remove OpenCut config directory (~/.opencut)
    ConfigDir := ExpandConstant('{userappdata}\..\..\.opencut');
    if DirExists(ConfigDir) then
      DelTree(ConfigDir, True, True, True);

    // Remove startup shortcut (in case autostart was selected)
    StartupShortcut := ExpandConstant('{userstartup}\OpenCut.lnk');
    if FileExists(StartupShortcut) then
      DeleteFile(StartupShortcut);

    // Remove desktop shortcut
    if FileExists(ExpandConstant('{autodesktop}\OpenCut.lnk')) then
      DeleteFile(ExpandConstant('{autodesktop}\OpenCut.lnk'));

    // Broadcast environment change so PATH update takes effect immediately
    Exec('cmd.exe', '/c setx OPENCUT_UNINSTALLED ""', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  end;
end;

[UninstallDelete]
; Remove entire install directory and all contents
Type: filesandordirs; Name: "{app}"
; CEP extension (redundant with code above, but belt-and-suspenders)
Type: filesandordirs; Name: "{userappdata}\Adobe\CEP\extensions\com.opencut.panel"
; Config directory
Type: filesandordirs; Name: "{%USERPROFILE}\.opencut"
; Logs
Type: filesandordirs; Name: "{app}\logs"
; FFmpeg
Type: filesandordirs; Name: "{app}\ffmpeg"
; Server
Type: filesandordirs; Name: "{app}\server"
; Extension copy
Type: filesandordirs; Name: "{app}\extension"
; Desktop shortcut
Type: files; Name: "{autodesktop}\OpenCut.lnk"
; Startup shortcut
Type: files; Name: "{userstartup}\OpenCut.lnk"

[UninstallRun]
; Kill server before uninstall files are removed
Filename: "taskkill.exe"; Parameters: "/F /IM OpenCut-Server.exe"; Flags: runhidden; RunOnceId: "KillServer"
