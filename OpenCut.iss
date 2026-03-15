; OpenCut Installer Script for Inno Setup 6
; Fully self-contained installer — bundles server exe, ffmpeg, and CEP extension

#define MyAppName "OpenCut"
#define MyAppVersion "1.2.0"
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

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "autostart"; Description: "Start OpenCut server when Windows starts"; GroupDescription: "Startup:"
Name: "installextension"; Description: "Install Adobe Premiere Pro CEP extension"; GroupDescription: "Adobe Integration:"; Flags: checkedonce
Name: "downloadmodel"; Description: "Download Whisper AI model for captions (~150MB, requires internet)"; GroupDescription: "AI Models:"; Flags: checkedonce
Name: "downloadmodel\tiny"; Description: "tiny (75MB) — Fastest, lower accuracy"; GroupDescription: "AI Models:"; Flags: exclusive unchecked
Name: "downloadmodel\base"; Description: "base (150MB) — Good balance (recommended)"; GroupDescription: "AI Models:"; Flags: exclusive
Name: "downloadmodel\small"; Description: "small (500MB) — Better accuracy, slower"; GroupDescription: "AI Models:"; Flags: exclusive unchecked
Name: "downloadmodel\medium"; Description: "medium (1.5GB) — High accuracy, requires more RAM"; GroupDescription: "AI Models:"; Flags: exclusive unchecked
Name: "downloadmodel\turbo"; Description: "turbo (1.6GB) — Best speed/accuracy ratio, needs good hardware"; GroupDescription: "AI Models:"; Flags: exclusive unchecked

[Files]
; Icon
Source: "img\logo.ico"; DestDir: "{app}"; Flags: ignoreversion

; Bundled server (PyInstaller output — includes Python runtime + all deps)
Source: "dist\OpenCut-Server\*"; DestDir: "{app}\server"; Flags: ignoreversion recursesubdirs createallsubdirs

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

// Copy CEP extension to Adobe's extensions folder
procedure InstallCEPExtension();
var
  ExtSrc, ExtDest, ExtParent: string;
begin
  ExtSrc := ExpandConstant('{app}\extension\com.opencut.panel');
  ExtParent := ExpandConstant('{userappdata}\Adobe\CEP\extensions');
  ExtDest := ExtParent + '\com.opencut.panel';

  // Create parent dir
  if not DirExists(ExtParent) then
    ForceDirectories(ExtParent);

  // Remove old extension
  if DirExists(ExtDest) then
    DelTree(ExtDest, True, True, True);

  // Copy recursively
  DirectoryCopy(ExtSrc, ExtDest);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    if WizardIsTaskSelected('installextension') then
      InstallCEPExtension();
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  ExtPath: string;
begin
  if CurUninstallStep = usUninstall then
  begin
    ExtPath := ExpandConstant('{userappdata}\Adobe\CEP\extensions\com.opencut.panel');
    if DirExists(ExtPath) then
      DelTree(ExtPath, True, True, True);
  end;
end;

[UninstallDelete]
Type: filesandordirs; Name: "{userappdata}\Adobe\CEP\extensions\com.opencut.panel"
Type: filesandordirs; Name: "{app}\logs"
