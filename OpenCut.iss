; OpenCut Installer Script for Inno Setup 6
; Fully self-contained installer — bundles server exe, ffmpeg, and CEP extension

#define MyAppName "OpenCut"
#define MyAppVersion "1.39.0"
#define MyAppPublisher "SysAdminDoc"
#define MyAppURL "https://github.com/SysAdminDoc/OpenCut"
#define BundledFfmpegVersion "8.1.2-essentials_build-www.gyan.dev"
#define BundledFfprobeVersion "8.1.2-essentials_build-www.gyan.dev"
#define BundledFfmpegSecurityFloor "release>=8.1.2 OR git-master>=2026-06-10 (commit b29bdd3715)"
#define BundledFfmpegSecurityCve "CVE-2026-8461"
#define BundledFfmpegFixCommit1 "374b726ffa878ee1cadb987bd1e1e20cc7ed8845"
#define BundledFfmpegFixCommit2 "5806e8b9f34f1b0663b3017ef9dd1aa5d08116d1"

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
Source: "installer\Export-OpenCutUserData.ps1"; DestDir: "{app}"; Flags: ignoreversion

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

; Enable unsigned CEP extensions (PlayerDebugMode). Cover CSXS 7 (CC 2014)
; through 18 (PPro 2025+) so modern Premiere installs (CSXS 13+) actually
; load the panel — the previous 7-12 range silently dropped support for
; CC 2023+ users.
; PlayerDebugMode keys are written via [Run] with runasoriginaluser so
; they land in the invoking user's hive, not the elevated admin's.
; (PrivilegesRequired=admin causes HKCU to resolve to the wrong user.)

[Run]
; Enable CEP PlayerDebugMode in the invoking user's HKCU (not the elevated admin)
Filename: "reg.exe"; Parameters: "add ""HKCU\Software\Adobe\CSXS.7""  /v PlayerDebugMode /t REG_SZ /d 1 /f"; Flags: runasoriginaluser runhidden; Tasks: installextension
Filename: "reg.exe"; Parameters: "add ""HKCU\Software\Adobe\CSXS.8""  /v PlayerDebugMode /t REG_SZ /d 1 /f"; Flags: runasoriginaluser runhidden; Tasks: installextension
Filename: "reg.exe"; Parameters: "add ""HKCU\Software\Adobe\CSXS.9""  /v PlayerDebugMode /t REG_SZ /d 1 /f"; Flags: runasoriginaluser runhidden; Tasks: installextension
Filename: "reg.exe"; Parameters: "add ""HKCU\Software\Adobe\CSXS.10"" /v PlayerDebugMode /t REG_SZ /d 1 /f"; Flags: runasoriginaluser runhidden; Tasks: installextension
Filename: "reg.exe"; Parameters: "add ""HKCU\Software\Adobe\CSXS.11"" /v PlayerDebugMode /t REG_SZ /d 1 /f"; Flags: runasoriginaluser runhidden; Tasks: installextension
Filename: "reg.exe"; Parameters: "add ""HKCU\Software\Adobe\CSXS.12"" /v PlayerDebugMode /t REG_SZ /d 1 /f"; Flags: runasoriginaluser runhidden; Tasks: installextension
Filename: "reg.exe"; Parameters: "add ""HKCU\Software\Adobe\CSXS.13"" /v PlayerDebugMode /t REG_SZ /d 1 /f"; Flags: runasoriginaluser runhidden; Tasks: installextension
Filename: "reg.exe"; Parameters: "add ""HKCU\Software\Adobe\CSXS.14"" /v PlayerDebugMode /t REG_SZ /d 1 /f"; Flags: runasoriginaluser runhidden; Tasks: installextension
Filename: "reg.exe"; Parameters: "add ""HKCU\Software\Adobe\CSXS.15"" /v PlayerDebugMode /t REG_SZ /d 1 /f"; Flags: runasoriginaluser runhidden; Tasks: installextension
Filename: "reg.exe"; Parameters: "add ""HKCU\Software\Adobe\CSXS.16"" /v PlayerDebugMode /t REG_SZ /d 1 /f"; Flags: runasoriginaluser runhidden; Tasks: installextension
Filename: "reg.exe"; Parameters: "add ""HKCU\Software\Adobe\CSXS.17"" /v PlayerDebugMode /t REG_SZ /d 1 /f"; Flags: runasoriginaluser runhidden; Tasks: installextension
Filename: "reg.exe"; Parameters: "add ""HKCU\Software\Adobe\CSXS.18"" /v PlayerDebugMode /t REG_SZ /d 1 /f"; Flags: runasoriginaluser runhidden; Tasks: installextension
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
  WM_SETTINGCHANGE = $001A;
  SMTO_ABORTIFHUNG = $0002;

var
  InteractiveRemoveUserData: Boolean;

// Broadcast WM_SETTINGCHANGE so running apps pick up environment changes
// without a logoff. Setup broadcasts automatically (ChangesEnvironment=yes)
// but the uninstaller does not, so RemoveFromPath needs this explicitly.
function SendMessageTimeout(Wnd: HWND; Msg: LongInt; wParam: LongInt;
  lParam: string; fuFlags: LongInt; uTimeout: LongInt;
  var lpdwResult: LongInt): LongInt;
  external 'SendMessageTimeoutW@user32.dll stdcall';

procedure RefreshEnvironment();
var
  MsgResult: LongInt;
begin
  SendMessageTimeout(HWND_BROADCAST, WM_SETTINGCHANGE, 0, 'Environment',
    SMTO_ABORTIFHUNG, 5000, MsgResult);
end;

// --- Kill OpenCut server processes ---

procedure KillOpenCutProcesses();
var
  ResultCode: Integer;
begin
  // Kill all OpenCut-Server.exe processes
  Exec('taskkill.exe', '/F /IM OpenCut-Server.exe', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  // Kill any wscript running our launcher (by image name, NOT window title — window title match can kill explorer.exe)
  Exec('taskkill.exe', '/F /IM wscript.exe /FI "WINDOWTITLE eq OpenCut-Launcher"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  // Kill python server if running from source (port 5679)
  Exec('cmd.exe', '/c for /f "tokens=5" %a in (''netstat -ano ^| findstr :5679 ^| findstr LISTENING'') do taskkill /F /PID %a', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  // Brief pause to let processes die
  Sleep(500);
end;

// --- PATH management for bundled FFmpeg ---

// True when Dir matches a full semicolon-delimited segment of PathValue
// (case-insensitive, trimmed). Substring matching would false-positive on
// sibling directories and corrupt unrelated PATH entries on removal.
function PathContainsDir(const PathValue, Dir: string): Boolean;
var
  Remaining, Segment: string;
  P: Integer;
begin
  Result := False;
  Remaining := PathValue;
  while Remaining <> '' do
  begin
    P := Pos(';', Remaining);
    if P > 0 then
    begin
      Segment := Copy(Remaining, 1, P - 1);
      Delete(Remaining, 1, P);
    end
    else
    begin
      Segment := Remaining;
      Remaining := '';
    end;
    if CompareText(Trim(Segment), Trim(Dir)) = 0 then
    begin
      Result := True;
      Exit;
    end;
  end;
end;

procedure AddToPath(Dir: string);
var
  OldPath: string;
begin
  if not RegQueryStringValue(HKCU, EnvironmentKey, 'Path', OldPath) then
    OldPath := '';
  if not PathContainsDir(OldPath, Dir) then
  begin
    if OldPath <> '' then
      OldPath := OldPath + ';';
    // Write REG_EXPAND_SZ: the user Path is commonly REG_EXPAND_SZ and
    // RegWriteStringValue would silently rewrite it as REG_SZ, breaking
    // every existing %VAR% entry. REG_EXPAND_SZ is also safe when the
    // original value was plain REG_SZ (literal paths expand to themselves).
    RegWriteExpandStringValue(HKCU, EnvironmentKey, 'Path', OldPath + Dir);
  end;
end;

function JsonEscape(Value: string): string;
begin
  Result := Value;
  StringChangeEx(Result, '\', '\\', True);
  StringChangeEx(Result, '"', '\"', True);
end;

procedure WriteInstallerManifest();
var
  ManifestDir, ManifestPath, Json: string;
begin
  ManifestDir := ExpandConstant('{%USERPROFILE}\.opencut');
  ManifestPath := ManifestDir + '\installer.json';
  ForceDirectories(ManifestDir);
  Json :=
    '{' + #13#10 +
    '  "app_name": "' + JsonEscape('{#MyAppName}') + '",' + #13#10 +
    '  "app_version": "' + JsonEscape('{#MyAppVersion}') + '",' + #13#10 +
    '  "installer_kind": "inno",' + #13#10 +
    '  "install_path": "' + JsonEscape(ExpandConstant('{app}')) + '",' + #13#10 +
    '  "server_path": "' + JsonEscape(ExpandConstant('{app}\server')) + '",' + #13#10 +
    '  "ffmpeg_path": "' + JsonEscape(ExpandConstant('{app}\ffmpeg')) + '",' + #13#10 +
    '  "bundled_ffmpeg_version": "' + JsonEscape('{#BundledFfmpegVersion}') + '",' + #13#10 +
    '  "bundled_ffprobe_version": "' + JsonEscape('{#BundledFfprobeVersion}') + '",' + #13#10 +
    '  "bundled_ffmpeg_security_floor": "' + JsonEscape('{#BundledFfmpegSecurityFloor}') + '",' + #13#10 +
    '  "bundled_ffmpeg_security_cve": "' + JsonEscape('{#BundledFfmpegSecurityCve}') + '",' + #13#10 +
    '  "bundled_ffmpeg_security_fix_commits": ["' +
      JsonEscape('{#BundledFfmpegFixCommit1}') + '", "' +
      JsonEscape('{#BundledFfmpegFixCommit2}') + '"]' + #13#10 +
    '}' + #13#10;
  SaveStringToFile(ManifestPath, Json, False);
end;

procedure VerifyInstalledMediaBinary(BinaryPath, DisplayName: string);
var
  ResultCode: Integer;
  Output: TExecOutput;
  Banner: string;
begin
  if not FileExists(BinaryPath) then
    RaiseException(DisplayName + ' is missing from the installer payload.');

  if not ExecAndCaptureOutput(BinaryPath, '-version', '', SW_HIDE,
      ewWaitUntilTerminated, ResultCode, Output) or
     (ResultCode <> 0) or Output.Error or
     (GetArrayLength(Output.StdOut) = 0) then
    RaiseException(DisplayName + ' version verification failed.');

  Banner := Output.StdOut[0];
  if Pos('version 8.1.2', Lowercase(Banner)) = 0 then
    RaiseException(
      DisplayName + ' is below the {#BundledFfmpegSecurityFloor} floor for ' +
      '{#BundledFfmpegSecurityCve}: ' + Banner);
  Log(DisplayName + ' security floor verified: ' + Banner);
end;

procedure VerifyInstalledFfmpeg();
begin
  VerifyInstalledMediaBinary(ExpandConstant('{app}\ffmpeg\ffmpeg.exe'), 'FFmpeg');
  VerifyInstalledMediaBinary(ExpandConstant('{app}\ffmpeg\ffprobe.exe'), 'FFprobe');
end;

procedure RemoveFromPath(Dir: string);
var
  OldPath, NewPath, Remaining, Segment: string;
  P: Integer;
  Removed: Boolean;
begin
  if not RegQueryStringValue(HKCU, EnvironmentKey, 'Path', OldPath) then
    Exit;
  // Rebuild the value from full semicolon-delimited segments so only exact
  // matches are dropped — the old substring Delete() could truncate an
  // unrelated longer entry (e.g. "...\OpenCut\ffmpeg-extras").
  NewPath := '';
  Removed := False;
  Remaining := OldPath;
  while Remaining <> '' do
  begin
    P := Pos(';', Remaining);
    if P > 0 then
    begin
      Segment := Copy(Remaining, 1, P - 1);
      Delete(Remaining, 1, P);
    end
    else
    begin
      Segment := Remaining;
      Remaining := '';
    end;
    if CompareText(Trim(Segment), Trim(Dir)) = 0 then
      Removed := True
    else if Trim(Segment) <> '' then
    begin
      if NewPath <> '' then
        NewPath := NewPath + ';';
      NewPath := NewPath + Segment;
    end;
  end;
  if Removed then
    // REG_EXPAND_SZ — see AddToPath. Never downgrade the value type.
    RegWriteExpandStringValue(HKCU, EnvironmentKey, 'Path', NewPath);
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
    VerifyInstalledFfmpeg();
    // Add bundled FFmpeg to user PATH
    AddToPath(ExpandConstant('{app}\ffmpeg'));
    // Write machine-readable installer manifest for support/debug tooling
    WriteInstallerManifest();
    // Install CEP extension
    if WizardIsTaskSelected('installextension') then
      InstallCEPExtension();
  end;
end;

// --- Uninstall hooks ---

function HasUninstallParameter(Name: string): Boolean;
var
  I: Integer;
begin
  Result := False;
  for I := 1 to ParamCount do
    if CompareText(ParamStr(I), Name) = 0 then
    begin
      Result := True;
      Exit;
    end;
end;

function GetUninstallParameter(Prefix, DefaultValue: string): string;
var
  I: Integer;
  Value: string;
begin
  Result := DefaultValue;
  for I := 1 to ParamCount do
  begin
    Value := ParamStr(I);
    if Pos(Uppercase(Prefix), Uppercase(Value)) = 1 then
    begin
      Result := Copy(Value, Length(Prefix) + 1, MaxInt);
      Exit;
    end;
  end;
end;

function InitializeUninstall(): Boolean;
var
  OptionsForm: TSetupForm;
  Heading, Detail: TNewStaticText;
  RemoveCheckBox: TNewCheckBox;
  ContinueButton, CancelButton: TNewButton;
begin
  InteractiveRemoveUserData := False;
  if UninstallSilent then
  begin
    Result := True;
    Exit;
  end;

  OptionsForm := CreateCustomForm(ScaleX(560), ScaleY(190), False, True);
  try
    OptionsForm.Caption := 'OpenCut uninstall options';
    OptionsForm.ClientWidth := ScaleX(560);
    OptionsForm.ClientHeight := ScaleY(190);

    Heading := TNewStaticText.Create(OptionsForm);
    Heading.Parent := OptionsForm;
    Heading.Left := ScaleX(20);
    Heading.Top := ScaleY(18);
    Heading.Caption := 'OpenCut user data is preserved by default.';
    Heading.Font.Style := [fsBold];

    Detail := TNewStaticText.Create(OptionsForm);
    Detail.Parent := OptionsForm;
    Detail.Left := ScaleX(20);
    Detail.Top := ScaleY(46);
    Detail.Width := ScaleX(520);
    Detail.Height := ScaleY(42);
    Detail.AutoSize := False;
    Detail.WordWrap := True;
    Detail.Caption :=
      'Settings, jobs, journals, indexes, plugins, models, and project/agent state remain in ' +
      GetUninstallParameter('/USERDATADIR=', ExpandConstant('{%USERPROFILE}\.opencut')) + '.';

    RemoveCheckBox := TNewCheckBox.Create(OptionsForm);
    RemoveCheckBox.Parent := OptionsForm;
    RemoveCheckBox.Left := ScaleX(20);
    RemoveCheckBox.Top := ScaleY(94);
    RemoveCheckBox.Width := ScaleX(520);
    RemoveCheckBox.Caption := 'Also remove this data after creating and validating a backup';
    RemoveCheckBox.Checked := False;

    ContinueButton := TNewButton.Create(OptionsForm);
    ContinueButton.Parent := OptionsForm;
    ContinueButton.Width := ScaleX(100);
    ContinueButton.Height := ScaleY(28);
    ContinueButton.Left := OptionsForm.ClientWidth - ScaleX(220);
    ContinueButton.Top := ScaleY(145);
    ContinueButton.Caption := 'Continue';
    ContinueButton.ModalResult := mrOk;
    ContinueButton.Default := True;

    CancelButton := TNewButton.Create(OptionsForm);
    CancelButton.Parent := OptionsForm;
    CancelButton.Width := ScaleX(100);
    CancelButton.Height := ScaleY(28);
    CancelButton.Left := OptionsForm.ClientWidth - ScaleX(110);
    CancelButton.Top := ScaleY(145);
    CancelButton.Caption := 'Cancel';
    CancelButton.ModalResult := mrCancel;
    CancelButton.Cancel := True;

    Result := OptionsForm.ShowModal() = mrOk;
    if Result then
      InteractiveRemoveUserData := RemoveCheckBox.Checked;
  finally
    OptionsForm.Free();
  end;
end;

function RemoveUserDataRequested(): Boolean;
begin
  Result := HasUninstallParameter('/REMOVEUSERDATA');
  if InteractiveRemoveUserData then
    Result := True;
end;

function BackupAndRemoveUserData(ConfigDir: string): Boolean;
var
  BackupDir, ScriptPath, ResultFile, Params: string;
  ResultLines: TArrayOfString;
  ResultCode: Integer;
begin
  Result := False;
  BackupDir := GetUninstallParameter(
    '/USERDATABACKUPDIR=', ExpandConstant('{userdocs}\OpenCut Backups'));
  ScriptPath := ExpandConstant('{app}\Export-OpenCutUserData.ps1');
  ResultFile := ExpandConstant('{tmp}\OpenCut-uninstall-data-result.txt');
  DeleteFile(ResultFile);

  Params :=
    '-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' + ScriptPath + '"' +
    ' -UserDataPath "' + ConfigDir + '"' +
    ' -BackupDirectory "' + BackupDir + '"' +
    ' -ResultFile "' + ResultFile + '" -RemoveAfterBackup';

  if Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'), Params,
      '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0) then
  begin
    if LoadStringsFromFile(ResultFile, ResultLines) and (GetArrayLength(ResultLines) > 0) then
      Log('OpenCut user-data removal result: ' + Trim(ResultLines[0]));
    Result := True;
  end
  else
  begin
    Log('OpenCut user data was preserved because backup/removal failed with code ' +
      IntToStr(ResultCode) + '.');
    if not UninstallSilent then
      MsgBox(
        'OpenCut user data was preserved because a verified backup could not be created or removal could not finish. ' +
        'The application uninstall will continue.',
        mbError, MB_OK);
  end;

  DeleteFile(ResultFile);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  ExtPath, ConfigDir, StartupShortcut: string;
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

    // Preserve user data by default. Explicit removal always creates and
    // validates a backup outside the deletion root before any delete attempt.
    ConfigDir := GetUninstallParameter(
      '/USERDATADIR=', ExpandConstant('{%USERPROFILE}\.opencut'));
    if RemoveUserDataRequested() and DirExists(ConfigDir) then
    begin
      if UninstallSilent or
         (MsgBox(
            'Remove all OpenCut user data after a verified backup?' + #13#10 + #13#10 +
            'Path: ' + ConfigDir + #13#10 +
            'Includes settings, jobs, journals, indexes, plugins, models, and project/agent state.',
            mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES) then
        BackupAndRemoveUserData(ConfigDir);
    end
    else
      Log('Preserving OpenCut user data at ' + ConfigDir + '.');

    // Remove startup shortcut (in case autostart was selected)
    StartupShortcut := ExpandConstant('{userstartup}\OpenCut.lnk');
    if FileExists(StartupShortcut) then
      DeleteFile(StartupShortcut);

    // Remove desktop shortcut
    if FileExists(ExpandConstant('{autodesktop}\OpenCut.lnk')) then
      DeleteFile(ExpandConstant('{autodesktop}\OpenCut.lnk'));

    // Broadcast environment change so the PATH removal takes effect
    // immediately. (The previous `setx OPENCUT_UNINSTALLED ""` hack left a
    // permanent stray env var behind and setx rejects empty values on some
    // Windows builds.)
    RefreshEnvironment();
  end;
end;

[UninstallDelete]
; Remove entire install directory and all contents
Type: filesandordirs; Name: "{app}"
; CEP extension (redundant with code above, but belt-and-suspenders)
Type: filesandordirs; Name: "{userappdata}\Adobe\CEP\extensions\com.opencut.panel"
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
