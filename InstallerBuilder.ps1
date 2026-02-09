<#
.SYNOPSIS
    OpenCut Installer Builder GUI
.DESCRIPTION
    GUI tool to build the OpenCut installer with verbose logging and diagnostics.
#>

Add-Type -AssemblyName PresentationFramework
Add-Type -AssemblyName PresentationCore
Add-Type -AssemblyName WindowsBase
Add-Type -AssemblyName System.Windows.Forms

$ErrorActionPreference = "Continue"

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $ScriptDir) { $ScriptDir = Get-Location }

$XAML = @"
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        Title="OpenCut Installer Builder" 
        Height="700" Width="900"
        WindowStartupLocation="CenterScreen"
        Background="#1a1a2e">
    <Window.Resources>
        <Style TargetType="Button">
            <Setter Property="Background" Value="#0f3460"/>
            <Setter Property="Foreground" Value="#00fff5"/>
            <Setter Property="BorderBrush" Value="#00fff5"/>
            <Setter Property="BorderThickness" Value="1"/>
            <Setter Property="Padding" Value="15,8"/>
            <Setter Property="FontSize" Value="13"/>
            <Setter Property="Cursor" Value="Hand"/>
            <Setter Property="Template">
                <Setter.Value>
                    <ControlTemplate TargetType="Button">
                        <Border Background="{TemplateBinding Background}" 
                                BorderBrush="{TemplateBinding BorderBrush}" 
                                BorderThickness="{TemplateBinding BorderThickness}"
                                CornerRadius="4"
                                Padding="{TemplateBinding Padding}">
                            <ContentPresenter HorizontalAlignment="Center" VerticalAlignment="Center"/>
                        </Border>
                        <ControlTemplate.Triggers>
                            <Trigger Property="IsMouseOver" Value="True">
                                <Setter Property="Background" Value="#16213e"/>
                                <Setter Property="BorderBrush" Value="#00fff5"/>
                            </Trigger>
                            <Trigger Property="IsEnabled" Value="False">
                                <Setter Property="Background" Value="#2a2a3e"/>
                                <Setter Property="Foreground" Value="#666"/>
                                <Setter Property="BorderBrush" Value="#444"/>
                            </Trigger>
                        </ControlTemplate.Triggers>
                    </ControlTemplate>
                </Setter.Value>
            </Setter>
        </Style>
        <Style TargetType="TextBox">
            <Setter Property="Background" Value="#16213e"/>
            <Setter Property="Foreground" Value="#e0e0e0"/>
            <Setter Property="BorderBrush" Value="#0f3460"/>
            <Setter Property="BorderThickness" Value="1"/>
            <Setter Property="Padding" Value="8"/>
            <Setter Property="FontFamily" Value="Consolas"/>
            <Setter Property="FontSize" Value="11"/>
        </Style>
        <Style TargetType="Label">
            <Setter Property="Foreground" Value="#e0e0e0"/>
            <Setter Property="FontSize" Value="12"/>
        </Style>
        <Style TargetType="CheckBox">
            <Setter Property="Foreground" Value="#e0e0e0"/>
            <Setter Property="FontSize" Value="12"/>
        </Style>
    </Window.Resources>
    
    <Grid Margin="20">
        <Grid.RowDefinitions>
            <RowDefinition Height="Auto"/>
            <RowDefinition Height="Auto"/>
            <RowDefinition Height="Auto"/>
            <RowDefinition Height="*"/>
            <RowDefinition Height="Auto"/>
        </Grid.RowDefinitions>
        
        <!-- Header -->
        <StackPanel Grid.Row="0" Margin="0,0,0,15">
            <TextBlock Text="OpenCut Installer Builder" 
                       FontSize="24" FontWeight="Bold" 
                       Foreground="#00fff5"/>
            <TextBlock Text="Build standalone installer with verbose diagnostics" 
                       FontSize="12" Foreground="#888" Margin="0,5,0,0"/>
        </StackPanel>
        
        <!-- Diagnostics Panel -->
        <Border Grid.Row="1" Background="#16213e" CornerRadius="6" Padding="15" Margin="0,0,0,15">
            <Grid>
                <Grid.ColumnDefinitions>
                    <ColumnDefinition Width="*"/>
                    <ColumnDefinition Width="*"/>
                </Grid.ColumnDefinitions>
                <Grid.RowDefinitions>
                    <RowDefinition Height="Auto"/>
                    <RowDefinition Height="Auto"/>
                    <RowDefinition Height="Auto"/>
                    <RowDefinition Height="Auto"/>
                    <RowDefinition Height="Auto"/>
                    <RowDefinition Height="Auto"/>
                </Grid.RowDefinitions>
                
                <TextBlock Grid.Row="0" Grid.ColumnSpan="2" Text="System Diagnostics" 
                           FontSize="14" FontWeight="Bold" Foreground="#00fff5" Margin="0,0,0,10"/>
                
                <StackPanel Grid.Row="1" Grid.Column="0" Orientation="Horizontal" Margin="0,3">
                    <TextBlock x:Name="chkInnoSetup" Text="[?]" Width="25" Foreground="#888"/>
                    <TextBlock Text="Inno Setup 6" Foreground="#e0e0e0"/>
                </StackPanel>
                <TextBlock x:Name="txtInnoPath" Grid.Row="1" Grid.Column="1" 
                           Text="Checking..." Foreground="#888" FontSize="10" 
                           VerticalAlignment="Center" TextTrimming="CharacterEllipsis"/>
                
                <StackPanel Grid.Row="2" Grid.Column="0" Orientation="Horizontal" Margin="0,3">
                    <TextBlock x:Name="chkOpencut" Text="[?]" Width="25" Foreground="#888"/>
                    <TextBlock Text="OpenCut Backend" Foreground="#e0e0e0"/>
                </StackPanel>
                <TextBlock x:Name="txtOpencutPath" Grid.Row="2" Grid.Column="1" 
                           Text="Checking..." Foreground="#888" FontSize="10"
                           VerticalAlignment="Center" TextTrimming="CharacterEllipsis"/>
                
                <StackPanel Grid.Row="3" Grid.Column="0" Orientation="Horizontal" Margin="0,3">
                    <TextBlock x:Name="chkExtension" Text="[?]" Width="25" Foreground="#888"/>
                    <TextBlock Text="CEP Extension" Foreground="#e0e0e0"/>
                </StackPanel>
                <TextBlock x:Name="txtExtensionPath" Grid.Row="3" Grid.Column="1" 
                           Text="Checking..." Foreground="#888" FontSize="10"
                           VerticalAlignment="Center" TextTrimming="CharacterEllipsis"/>
                
                <StackPanel Grid.Row="4" Grid.Column="0" Orientation="Horizontal" Margin="0,3">
                    <TextBlock x:Name="chkLauncher" Text="[?]" Width="25" Foreground="#888"/>
                    <TextBlock Text="Launcher Script" Foreground="#e0e0e0"/>
                </StackPanel>
                <TextBlock x:Name="txtLauncherPath" Grid.Row="4" Grid.Column="1" 
                           Text="Checking..." Foreground="#888" FontSize="10"
                           VerticalAlignment="Center" TextTrimming="CharacterEllipsis"/>
                
                <StackPanel Grid.Row="5" Grid.Column="0" Orientation="Horizontal" Margin="0,3">
                    <TextBlock x:Name="chkIssFile" Text="[?]" Width="25" Foreground="#888"/>
                    <TextBlock Text="ISS Script" Foreground="#e0e0e0"/>
                </StackPanel>
                <TextBlock x:Name="txtIssPath" Grid.Row="5" Grid.Column="1" 
                           Text="Checking..." Foreground="#888" FontSize="10"
                           VerticalAlignment="Center" TextTrimming="CharacterEllipsis"/>
            </Grid>
        </Border>
        
        <!-- Actions -->
        <StackPanel Grid.Row="2" Orientation="Horizontal" Margin="0,0,0,15">
            <Button x:Name="btnDiagnose" Content="Run Diagnostics" Margin="0,0,10,0"/>
            <Button x:Name="btnGenerate" Content="Generate ISS File" Margin="0,0,10,0"/>
            <Button x:Name="btnBuild" Content="Build Installer" Margin="0,0,10,0"/>
            <Button x:Name="btnOpenFolder" Content="Open Output Folder" Margin="0,0,10,0"/>
            <Button x:Name="btnClearLog" Content="Clear Log" Margin="0,0,0,0"/>
        </StackPanel>
        
        <!-- Log Output -->
        <Border Grid.Row="3" Background="#0d0d1a" CornerRadius="6" Padding="2">
            <TextBox x:Name="txtLog" 
                     IsReadOnly="True" 
                     VerticalScrollBarVisibility="Auto"
                     HorizontalScrollBarVisibility="Auto"
                     TextWrapping="NoWrap"
                     AcceptsReturn="True"
                     Background="#0d0d1a"
                     BorderThickness="0"/>
        </Border>
        
        <!-- Status Bar -->
        <Border Grid.Row="4" Background="#0f3460" CornerRadius="4" Padding="10,8" Margin="0,15,0,0">
            <Grid>
                <Grid.ColumnDefinitions>
                    <ColumnDefinition Width="*"/>
                    <ColumnDefinition Width="Auto"/>
                </Grid.ColumnDefinitions>
                <TextBlock x:Name="txtStatus" Text="Ready" Foreground="#00fff5" VerticalAlignment="Center"/>
                <TextBlock x:Name="txtTime" Grid.Column="1" Text="" Foreground="#888" VerticalAlignment="Center"/>
            </Grid>
        </Border>
    </Grid>
</Window>
"@

# Parse XAML
$reader = [System.Xml.XmlReader]::Create([System.IO.StringReader]::new($XAML))
$window = [Windows.Markup.XamlReader]::Load($reader)

# Get controls
$chkInnoSetup = $window.FindName("chkInnoSetup")
$txtInnoPath = $window.FindName("txtInnoPath")
$chkOpencut = $window.FindName("chkOpencut")
$txtOpencutPath = $window.FindName("txtOpencutPath")
$chkExtension = $window.FindName("chkExtension")
$txtExtensionPath = $window.FindName("txtExtensionPath")
$chkLauncher = $window.FindName("chkLauncher")
$txtLauncherPath = $window.FindName("txtLauncherPath")
$chkIssFile = $window.FindName("chkIssFile")
$txtIssPath = $window.FindName("txtIssPath")
$btnDiagnose = $window.FindName("btnDiagnose")
$btnGenerate = $window.FindName("btnGenerate")
$btnBuild = $window.FindName("btnBuild")
$btnOpenFolder = $window.FindName("btnOpenFolder")
$btnClearLog = $window.FindName("btnClearLog")
$txtLog = $window.FindName("txtLog")
$txtStatus = $window.FindName("txtStatus")
$txtTime = $window.FindName("txtTime")

# Logging function
function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )
    $timestamp = Get-Date -Format "HH:mm:ss.fff"
    $prefix = switch ($Level) {
        "INFO"    { "[INFO ]" }
        "SUCCESS" { "[OK   ]" }
        "ERROR"   { "[ERROR]" }
        "WARN"    { "[WARN ]" }
        "DEBUG"   { "[DEBUG]" }
        default   { "[     ]" }
    }
    $line = "$timestamp $prefix $Message`r`n"
    $window.Dispatcher.Invoke([Action]{
        $txtLog.AppendText($line)
        $txtLog.ScrollToEnd()
    })
}

function Set-Status {
    param([string]$Text)
    $window.Dispatcher.Invoke([Action]{
        $txtStatus.Text = $Text
        $txtTime.Text = Get-Date -Format "HH:mm:ss"
    })
}

function Set-Check {
    param(
        [System.Windows.Controls.TextBlock]$Control,
        [bool]$Success
    )
    $window.Dispatcher.Invoke([Action]{
        if ($Success) {
            $Control.Text = "[OK]"
            $Control.Foreground = [System.Windows.Media.Brushes]::LimeGreen
        } else {
            $Control.Text = "[X]"
            $Control.Foreground = [System.Windows.Media.Brushes]::Red
        }
    })
}

function Set-PathText {
    param(
        [System.Windows.Controls.TextBlock]$Control,
        [string]$Text,
        [bool]$Success
    )
    $window.Dispatcher.Invoke([Action]{
        $Control.Text = $Text
        if ($Success) {
            $Control.Foreground = [System.Windows.Media.Brushes]::LimeGreen
        } else {
            $Control.Foreground = [System.Windows.Media.Brushes]::OrangeRed
        }
    })
}

# Global variables
$script:InnoPath = $null
$script:OpencutPath = $null
$script:ExtensionPath = $null
$script:LauncherPath = $null
$script:IssPath = $null
$script:OutputDir = $null

# Diagnose function
function Run-Diagnostics {
    Write-Log "=" * 60
    Write-Log "Starting System Diagnostics"
    Write-Log "=" * 60
    Write-Log ""
    Write-Log "Script Directory: $ScriptDir" "DEBUG"
    Write-Log "Current Directory: $(Get-Location)" "DEBUG"
    Write-Log "PowerShell Version: $($PSVersionTable.PSVersion)" "DEBUG"
    Write-Log "OS: $([System.Environment]::OSVersion.VersionString)" "DEBUG"
    Write-Log ""
    
    Set-Status "Running diagnostics..."
    
    # Check Inno Setup
    Write-Log "Checking for Inno Setup 6..."
    $innoPaths = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 5\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 5\ISCC.exe"
    )
    
    $script:InnoPath = $null
    foreach ($path in $innoPaths) {
        Write-Log "  Checking: $path" "DEBUG"
        if (Test-Path $path) {
            $script:InnoPath = $path
            Write-Log "  FOUND: $path" "SUCCESS"
            break
        }
    }
    
    if ($script:InnoPath) {
        Set-Check $chkInnoSetup $true
        Set-PathText $txtInnoPath $script:InnoPath $true
        
        # Get version
        try {
            $versionInfo = (Get-Item $script:InnoPath).VersionInfo
            Write-Log "  Version: $($versionInfo.FileVersion)" "DEBUG"
        } catch {
            Write-Log "  Could not get version info" "WARN"
        }
    } else {
        Set-Check $chkInnoSetup $false
        Set-PathText $txtInnoPath "NOT FOUND - Install from jrsoftware.org/isdl.php" $false
        Write-Log "  Inno Setup NOT FOUND!" "ERROR"
    }
    Write-Log ""
    
    # Check OpenCut backend
    Write-Log "Checking for OpenCut backend..."
    $opencutPaths = @(
        (Join-Path $ScriptDir "opencut"),
        (Join-Path $ScriptDir "opencut\server.py"),
        (Join-Path $ScriptDir "opencut\__init__.py")
    )
    
    $script:OpencutPath = Join-Path $ScriptDir "opencut"
    Write-Log "  Expected path: $script:OpencutPath" "DEBUG"
    
    if (Test-Path $script:OpencutPath) {
        $serverPy = Join-Path $script:OpencutPath "server.py"
        Write-Log "  Directory exists: $script:OpencutPath" "DEBUG"
        Write-Log "  Checking for server.py: $serverPy" "DEBUG"
        
        if (Test-Path $serverPy) {
            Set-Check $chkOpencut $true
            Set-PathText $txtOpencutPath $script:OpencutPath $true
            Write-Log "  FOUND: $script:OpencutPath" "SUCCESS"
            
            # Count files
            $fileCount = (Get-ChildItem $script:OpencutPath -Recurse -File).Count
            Write-Log "  Files in opencut: $fileCount" "DEBUG"
        } else {
            Set-Check $chkOpencut $false
            Set-PathText $txtOpencutPath "server.py not found in opencut folder" $false
            Write-Log "  server.py NOT FOUND!" "ERROR"
        }
    } else {
        Set-Check $chkOpencut $false
        Set-PathText $txtOpencutPath "Folder not found: $script:OpencutPath" $false
        Write-Log "  opencut folder NOT FOUND!" "ERROR"
    }
    Write-Log ""
    
    # Check Extension
    Write-Log "Checking for CEP Extension..."
    $script:ExtensionPath = Join-Path $ScriptDir "extension\com.opencut.panel"
    Write-Log "  Expected path: $script:ExtensionPath" "DEBUG"
    
    if (Test-Path $script:ExtensionPath) {
        $manifestPath = Join-Path $script:ExtensionPath "CSXS\manifest.xml"
        Write-Log "  Directory exists" "DEBUG"
        Write-Log "  Checking manifest: $manifestPath" "DEBUG"
        
        if (Test-Path $manifestPath) {
            Set-Check $chkExtension $true
            Set-PathText $txtExtensionPath $script:ExtensionPath $true
            Write-Log "  FOUND: $script:ExtensionPath" "SUCCESS"
        } else {
            Set-Check $chkExtension $false
            Set-PathText $txtExtensionPath "manifest.xml not found" $false
            Write-Log "  manifest.xml NOT FOUND!" "ERROR"
        }
    } else {
        Set-Check $chkExtension $false
        Set-PathText $txtExtensionPath "Folder not found" $false
        Write-Log "  Extension folder NOT FOUND!" "ERROR"
    }
    Write-Log ""
    
    # Check Launcher
    Write-Log "Checking for Launcher scripts..."
    $script:LauncherPath = Join-Path $ScriptDir "OpenCut-Server.bat"
    $script:VbsLauncherPath = Join-Path $ScriptDir "OpenCut-Server.vbs"
    
    Write-Log "  BAT launcher: $script:LauncherPath" "DEBUG"
    Write-Log "  VBS launcher: $script:VbsLauncherPath" "DEBUG"
    
    $batExists = Test-Path $script:LauncherPath
    $vbsExists = Test-Path $script:VbsLauncherPath
    
    if ($batExists -and $vbsExists) {
        Set-Check $chkLauncher $true
        Set-PathText $txtLauncherPath "Both launchers present" $true
        Write-Log "  FOUND: Both launchers" "SUCCESS"
    } elseif ($batExists -or $vbsExists) {
        Set-Check $chkLauncher $true
        Set-PathText $txtLauncherPath "Partial - will create missing" $false
        Write-Log "  Partial - missing launchers will be created" "WARN"
    } else {
        Set-Check $chkLauncher $false
        Set-PathText $txtLauncherPath "Will be created" $false
        Write-Log "  Launchers NOT FOUND - will create" "WARN"
    }
    Write-Log ""
    
    # Check ISS file
    Write-Log "Checking for ISS file..."
    $script:IssPath = Join-Path $ScriptDir "OpenCut.iss"
    Write-Log "  Expected path: $script:IssPath" "DEBUG"
    
    if (Test-Path $script:IssPath) {
        Set-Check $chkIssFile $true
        Set-PathText $txtIssPath $script:IssPath $true
        Write-Log "  FOUND: $script:IssPath" "SUCCESS"
        
        # Check ISS content
        $issContent = Get-Content $script:IssPath -Raw
        Write-Log "  ISS file size: $($issContent.Length) bytes" "DEBUG"
    } else {
        Set-Check $chkIssFile $false
        Set-PathText $txtIssPath "Will be generated" $false
        Write-Log "  ISS file NOT FOUND - will generate" "WARN"
    }
    Write-Log ""
    
    # Check output directory
    $script:OutputDir = Join-Path $ScriptDir "installer\dist"
    Write-Log "Output directory: $script:OutputDir" "DEBUG"
    if (-not (Test-Path $script:OutputDir)) {
        Write-Log "  Creating output directory..." "DEBUG"
        New-Item -ItemType Directory -Path $script:OutputDir -Force | Out-Null
    }
    
    # List directory contents
    Write-Log ""
    Write-Log "Directory Contents of $ScriptDir :"
    Get-ChildItem $ScriptDir | ForEach-Object {
        $type = if ($_.PSIsContainer) { "[DIR ]" } else { "[FILE]" }
        Write-Log "  $type $($_.Name)" "DEBUG"
    }
    
    Write-Log ""
    Write-Log "=" * 60
    Write-Log "Diagnostics Complete"
    Write-Log "=" * 60
    
    Set-Status "Diagnostics complete"
}

# Generate ISS file
function Generate-IssFile {
    Write-Log ""
    Write-Log "=" * 60
    Write-Log "Generating ISS File"
    Write-Log "=" * 60
    
    Set-Status "Generating ISS file..."
    
    # Create launcher scripts if missing
    $vbsPath = Join-Path $ScriptDir "OpenCut-Server.vbs"
    
    if (-not (Test-Path $vbsPath)) {
        Write-Log "Creating hidden launcher (VBS)..." "INFO"
        
        # Hidden launcher (VBS) - runs server with no visible window
        $vbsContent = @'
Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

strPath = fso.GetParentFolderName(WScript.ScriptFullName)

' Build environment
strPython = "python"
strEnv = "set OPENCUT_HOME=" & strPath

If fso.FileExists(strPath & "\python\python.exe") Then
    strPython = """" & strPath & "\python\python.exe"""
    strEnv = strEnv & " & set PATH=" & strPath & "\python;" & strPath & "\python\Scripts;%PATH%"
End If

If fso.FolderExists(strPath & "\ffmpeg") Then
    strEnv = strEnv & " & set PATH=" & strPath & "\ffmpeg;%PATH%"
End If

If fso.FolderExists(strPath & "\models") Then
    strEnv = strEnv & " & set OPENCUT_BUNDLED=true"
    strEnv = strEnv & " & set WHISPER_MODELS_DIR=" & strPath & "\models\whisper"
    strEnv = strEnv & " & set TORCH_HOME=" & strPath & "\models\demucs"
    strEnv = strEnv & " & set OPENCUT_FLORENCE_DIR=" & strPath & "\models\florence"
    strEnv = strEnv & " & set OPENCUT_LAMA_DIR=" & strPath & "\models\lama"
End If

strCmd = "cmd /c """ & strEnv & " & " & strPython & " -m opencut.server"""

' Run completely hidden (0 = hidden, False = don't wait)
WshShell.Run strCmd, 0, False
'@
        $vbsContent | Set-Content -Path $vbsPath -Encoding ASCII
        Write-Log "  Created: $vbsPath" "SUCCESS"
    } else {
        Write-Log "Hidden launcher already exists: $vbsPath" "DEBUG"
    }
    
    if (-not (Test-Path $script:LauncherPath)) {
        Write-Log "Creating debug launcher (BAT)..." "INFO"
        
        # Debug launcher (BAT) - visible console for troubleshooting
        $batContent = @'
@echo off
setlocal

set "OPENCUT_HOME=%~dp0"
set "OPENCUT_HOME=%OPENCUT_HOME:~0,-1%"

if exist "%OPENCUT_HOME%\python\python.exe" (
    set "PYTHON=%OPENCUT_HOME%\python\python.exe"
    set "PATH=%OPENCUT_HOME%\python;%OPENCUT_HOME%\python\Scripts;%PATH%"
) else (
    set "PYTHON=python"
)

if exist "%OPENCUT_HOME%\ffmpeg" set "PATH=%OPENCUT_HOME%\ffmpeg;%PATH%"

if exist "%OPENCUT_HOME%\models" (
    set "OPENCUT_BUNDLED=true"
    set "WHISPER_MODELS_DIR=%OPENCUT_HOME%\models\whisper"
    set "TORCH_HOME=%OPENCUT_HOME%\models\demucs"
    set "OPENCUT_FLORENCE_DIR=%OPENCUT_HOME%\models\florence"
    set "OPENCUT_LAMA_DIR=%OPENCUT_HOME%\models\lama"
)

echo  OpenCut Server - Debug Mode
echo  Close this window to stop the server
echo.

"%PYTHON%" -m opencut.server
pause
'@
        $batContent | Set-Content -Path $script:LauncherPath -Encoding ASCII
        Write-Log "  Created: $script:LauncherPath" "SUCCESS"
    } else {
        Write-Log "Debug launcher already exists: $script:LauncherPath" "DEBUG"
    }
    
    # Generate ISS content
    Write-Log "Generating ISS content..." "INFO"
    
    $vbsLauncherPath = Join-Path $ScriptDir "OpenCut-Server.vbs"
    
    $issContent = @"
; OpenCut Installer Script
; Generated by OpenCut Installer Builder
; $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

#define MyAppName "OpenCut"
#define MyAppVersion "0.6.5"
#define MyAppPublisher "OpenCut"
#define MyAppURL "https://github.com/opencut"

[Setup]
AppId={{8A7B9C0D-1E2F-3A4B-5C6D-7E8F9A0B1C2D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
OutputDir=$($script:OutputDir -replace '\\', '\\')
OutputBaseFilename=OpenCut-Setup-{#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "autostart"; Description: "Start OpenCut automatically when Windows starts"; GroupDescription: "Startup:"; Flags: checkedonce
Name: "installextension"; Description: "Install Adobe Premiere Pro Extension"; GroupDescription: "Adobe Integration:"; Flags: checkedonce

[Files]
; OpenCut backend
Source: "$($script:OpencutPath -replace '\\', '\\')\*"; DestDir: "{app}\opencut"; Flags: ignoreversion recursesubdirs createallsubdirs

; CEP Extension
Source: "$($script:ExtensionPath -replace '\\', '\\')\*"; DestDir: "{app}\extension\com.opencut.panel"; Flags: ignoreversion recursesubdirs createallsubdirs

; Hidden launcher (VBS) - runs server invisibly
Source: "$($vbsLauncherPath -replace '\\', '\\')"; DestDir: "{app}"; Flags: ignoreversion

; Debug launcher (BAT) - for troubleshooting only
Source: "$($script:LauncherPath -replace '\\', '\\')"; DestDir: "{app}"; Flags: ignoreversion

[Dirs]
Name: "{app}\logs"

[Icons]
; Start menu - uses hidden launcher
Name: "{group}\OpenCut"; Filename: "wscript.exe"; Parameters: """{app}\OpenCut-Server.vbs"""; WorkingDir: "{app}"
Name: "{group}\OpenCut (Debug Mode)"; Filename: "{app}\OpenCut-Server.bat"; WorkingDir: "{app}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
; Desktop shortcut - hidden launcher
Name: "{autodesktop}\OpenCut"; Filename: "wscript.exe"; Parameters: """{app}\OpenCut-Server.vbs"""; WorkingDir: "{app}"; Tasks: desktopicon
; Startup folder - autostart with Windows
Name: "{userstartup}\OpenCut"; Filename: "wscript.exe"; Parameters: """{app}\OpenCut-Server.vbs"""; WorkingDir: "{app}"; Tasks: autostart

[Run]
; Start server silently after install
Filename: "wscript.exe"; Parameters: """{app}\OpenCut-Server.vbs"""; Description: "Start OpenCut Server"; Flags: nowait postinstall skipifsilent

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
            Result := FileCopy(SourceFilePath, DestFilePath, False) and Result;
        end;
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
  end;
end;

procedure InstallCEPExtension();
var
  ExtSrc, ExtDest: string;
begin
  ExtSrc := ExpandConstant('{app}\extension\com.opencut.panel');
  ExtDest := ExpandConstant('{userappdata}\Adobe\CEP\extensions\com.opencut.panel');
  if not DirExists(ExpandConstant('{userappdata}\Adobe\CEP\extensions')) then
    CreateDir(ExpandConstant('{userappdata}\Adobe\CEP\extensions'));
  if DirExists(ExtSrc) then
    DirectoryCopy(ExtSrc, ExtDest);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    if IsTaskSelected('installextension') then
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
"@
    
    # Write ISS file
    Write-Log "Writing ISS file to: $script:IssPath" "INFO"
    $issContent | Set-Content -Path $script:IssPath -Encoding UTF8
    
    if (Test-Path $script:IssPath) {
        Set-Check $chkIssFile $true
        Set-PathText $txtIssPath $script:IssPath $true
        Write-Log "  ISS file created successfully" "SUCCESS"
        Write-Log "  File size: $((Get-Item $script:IssPath).Length) bytes" "DEBUG"
        
        # Show ISS content summary
        Write-Log ""
        Write-Log "ISS File Contents Preview:" "DEBUG"
        Write-Log "-" * 40
        $issContent.Split("`n") | Select-Object -First 30 | ForEach-Object {
            Write-Log "  $_" "DEBUG"
        }
        Write-Log "  ..." "DEBUG"
        Write-Log "-" * 40
    } else {
        Write-Log "  FAILED to create ISS file!" "ERROR"
    }
    
    Write-Log ""
    Write-Log "ISS Generation Complete"
    Set-Status "ISS file generated"
}

# Build installer
function Build-Installer {
    Write-Log ""
    Write-Log "=" * 60
    Write-Log "Building Installer"
    Write-Log "=" * 60
    
    Set-Status "Building installer..."
    
    # Verify prerequisites
    if (-not $script:InnoPath) {
        Write-Log "ERROR: Inno Setup not found!" "ERROR"
        Write-Log "Please install Inno Setup 6 from: https://jrsoftware.org/isdl.php" "ERROR"
        Set-Status "Build failed - Inno Setup not found"
        return
    }
    
    if (-not (Test-Path $script:IssPath)) {
        Write-Log "ISS file not found, generating..." "WARN"
        Generate-IssFile
    }
    
    if (-not (Test-Path $script:IssPath)) {
        Write-Log "ERROR: Could not create ISS file!" "ERROR"
        Set-Status "Build failed - ISS file error"
        return
    }
    
    # Verify source paths in ISS file
    Write-Log "Verifying source paths..." "INFO"
    
    $issContent = Get-Content $script:IssPath -Raw
    
    # Extract Source paths from ISS
    $sourcePattern = 'Source:\s*"([^"]+)"'
    $matches = [regex]::Matches($issContent, $sourcePattern)
    
    foreach ($match in $matches) {
        $sourcePath = $match.Groups[1].Value
        # Handle wildcards
        $checkPath = $sourcePath -replace '\\\*$', ''
        
        Write-Log "  Checking: $checkPath" "DEBUG"
        
        if (Test-Path $checkPath) {
            Write-Log "    EXISTS" "SUCCESS"
        } else {
            Write-Log "    NOT FOUND!" "ERROR"
        }
    }
    
    Write-Log ""
    Write-Log "Running Inno Setup Compiler..." "INFO"
    Write-Log "  Command: `"$script:InnoPath`" `"$script:IssPath`"" "DEBUG"
    Write-Log ""
    
    # Run ISCC
    $startTime = Get-Date
    
    try {
        $process = New-Object System.Diagnostics.Process
        $process.StartInfo.FileName = $script:InnoPath
        $process.StartInfo.Arguments = "`"$script:IssPath`""
        $process.StartInfo.UseShellExecute = $false
        $process.StartInfo.RedirectStandardOutput = $true
        $process.StartInfo.RedirectStandardError = $true
        $process.StartInfo.CreateNoWindow = $true
        $process.StartInfo.WorkingDirectory = $ScriptDir
        
        Write-Log "Starting ISCC process..." "DEBUG"
        
        $process.Start() | Out-Null
        
        # Read output
        while (-not $process.HasExited) {
            $line = $process.StandardOutput.ReadLine()
            if ($line) {
                Write-Log "  [ISCC] $line" "DEBUG"
            }
            Start-Sleep -Milliseconds 100
        }
        
        # Read remaining output
        $remainingOutput = $process.StandardOutput.ReadToEnd()
        $remainingError = $process.StandardError.ReadToEnd()
        
        if ($remainingOutput) {
            $remainingOutput.Split("`n") | ForEach-Object {
                if ($_.Trim()) { Write-Log "  [ISCC] $_" "DEBUG" }
            }
        }
        
        if ($remainingError) {
            $remainingError.Split("`n") | ForEach-Object {
                if ($_.Trim()) { Write-Log "  [ISCC ERROR] $_" "ERROR" }
            }
        }
        
        $exitCode = $process.ExitCode
        $duration = (Get-Date) - $startTime
        
        Write-Log ""
        Write-Log "ISCC Exit Code: $exitCode" $(if ($exitCode -eq 0) { "SUCCESS" } else { "ERROR" })
        Write-Log "Build Duration: $($duration.TotalSeconds.ToString('F1')) seconds" "DEBUG"
        
        if ($exitCode -eq 0) {
            $outputExe = Join-Path $script:OutputDir "OpenCut-Setup-0.6.5.exe"
            
            if (Test-Path $outputExe) {
                $fileSize = (Get-Item $outputExe).Length
                $fileSizeMB = [math]::Round($fileSize / 1MB, 2)
                
                Write-Log ""
                Write-Log "=" * 60
                Write-Log "BUILD SUCCESSFUL!" "SUCCESS"
                Write-Log "=" * 60
                Write-Log ""
                Write-Log "Output: $outputExe" "SUCCESS"
                Write-Log "Size: $fileSizeMB MB" "SUCCESS"
                
                Set-Status "Build successful! Output: $outputExe"
            } else {
                Write-Log "WARNING: Exit code 0 but output file not found!" "WARN"
                Write-Log "Expected: $outputExe" "WARN"
                
                # List output directory
                Write-Log ""
                Write-Log "Contents of output directory:" "DEBUG"
                if (Test-Path $script:OutputDir) {
                    Get-ChildItem $script:OutputDir | ForEach-Object {
                        Write-Log "  $($_.Name) - $($_.Length) bytes" "DEBUG"
                    }
                } else {
                    Write-Log "  Directory does not exist!" "ERROR"
                }
                
                Set-Status "Build completed but output not found"
            }
        } else {
            Write-Log ""
            Write-Log "BUILD FAILED!" "ERROR"
            Write-Log "Check the log above for error details." "ERROR"
            Set-Status "Build failed with exit code $exitCode"
        }
        
    } catch {
        Write-Log "Exception during build: $($_.Exception.Message)" "ERROR"
        Write-Log "Stack trace: $($_.ScriptStackTrace)" "ERROR"
        Set-Status "Build failed with exception"
    }
}

# Event handlers
$btnDiagnose.Add_Click({
    $txtLog.Clear()
    Run-Diagnostics
})

$btnGenerate.Add_Click({
    Generate-IssFile
})

$btnBuild.Add_Click({
    Build-Installer
})

$btnOpenFolder.Add_Click({
    $outputDir = Join-Path $ScriptDir "installer\dist"
    if (-not (Test-Path $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    }
    Start-Process explorer.exe -ArgumentList $outputDir
})

$btnClearLog.Add_Click({
    $txtLog.Clear()
})

# Initial diagnostics on load
$window.Add_Loaded({
    Write-Log "OpenCut Installer Builder v1.0"
    Write-Log "Script Location: $ScriptDir"
    Write-Log ""
    Run-Diagnostics
})

# Show window
$window.ShowDialog() | Out-Null
