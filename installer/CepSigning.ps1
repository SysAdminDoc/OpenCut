<#
.SYNOPSIS
    Sign and verify the OpenCut CEP panel with Adobe's ZXPSignCmd.
.DESCRIPTION
    Dot-source this file. InstallerBuilder.ps1 signs the staged panel with
    Invoke-CepPanelSigning, and the release smoke checks the installed copy
    with Assert-CepPanelSignature.

    Settings come from parameters first, then from the process or User
    environment:
      OPENCUT_CEP_CERT           path to the signing .p12
      OPENCUT_CEP_CERT_PASSWORD  its password
      OPENCUT_CEP_TSA_URL        timestamp server (default DigiCert)
      OPENCUT_ZXPSIGNCMD         path to ZXPSignCmd.exe (otherwise PATH, then
                                 %LOCALAPPDATA%\Programs\ZXPSignCmd\<version>)

    Keep this file ASCII. Windows PowerShell 5.1 reads a BOM-less script in
    the ANSI code page.
#>

$script:CepDefaultTimestampUrl = "http://timestamp.digicert.com"

# Names that must never be signed into the shipped panel. Adobe rejects
# .DS_Store and __MACOSX, META-INF and mimetype would be a stale signature, and
# .debug opens a CEF DevTools port that only source checkouts should have.
$script:CepForbiddenNames = @(".debug", ".DS_Store", "__MACOSX", "Thumbs.db", "META-INF", "mimetype")

function Get-CepSigningSetting {
    param([Parameter(Mandatory = $true)][string]$Name)
    $value = [Environment]::GetEnvironmentVariable($Name, "Process")
    if ([string]::IsNullOrEmpty($value)) {
        $value = [Environment]::GetEnvironmentVariable($Name, "User")
    }
    return $value
}

function Resolve-ZxpSignCmd {
    param([string]$Path = "")
    if ($Path) {
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
            throw "ZXPSignCmd not found at $Path"
        }
        return (Resolve-Path -LiteralPath $Path).Path
    }
    $candidates = New-Object System.Collections.Generic.List[string]
    $fromEnv = Get-CepSigningSetting "OPENCUT_ZXPSIGNCMD"
    if ($fromEnv) { $candidates.Add($fromEnv) }
    $onPath = Get-Command "ZXPSignCmd.exe" -ErrorAction SilentlyContinue
    if ($onPath) { $candidates.Add($onPath.Source) }
    if ($env:LOCALAPPDATA) {
        $programs = Join-Path $env:LOCALAPPDATA "Programs\ZXPSignCmd"
        if (Test-Path -LiteralPath $programs) {
            $versions = Get-ChildItem -LiteralPath $programs -Directory |
                Sort-Object -Descending -Property @{ Expression = {
                    $parsed = $null
                    if ([version]::TryParse($_.Name, [ref]$parsed)) { $parsed } else { [version]"0.0" }
                } }
            foreach ($dir in $versions) { $candidates.Add((Join-Path $dir.FullName "ZXPSignCmd.exe")) }
        }
    }
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    throw ("ZXPSignCmd.exe not found. Install the x64 build from " +
        "https://github.com/Adobe-CEP/CEP-Resources/tree/master/ZXPSignCMD into " +
        "%LOCALAPPDATA%\Programs\ZXPSignCmd\<version>\ or set OPENCUT_ZXPSIGNCMD.")
}

function Resolve-CepSigningInputs {
    param(
        [string]$Certificate = "",
        [string]$Password = "",
        [string]$TimestampUrl = "",
        [string]$ZxpSignCmd = ""
    )
    if (-not $Certificate) { $Certificate = Get-CepSigningSetting "OPENCUT_CEP_CERT" }
    if (-not $Certificate) {
        throw "OPENCUT_CEP_CERT is not set. Point it at the panel signing .p12 (DEVELOPMENT.md, Signing the panel)."
    }
    if (-not (Test-Path -LiteralPath $Certificate -PathType Leaf)) {
        throw "CEP signing certificate not found at $Certificate"
    }
    if (-not $Password) { $Password = Get-CepSigningSetting "OPENCUT_CEP_CERT_PASSWORD" }
    if (-not $Password) { throw "OPENCUT_CEP_CERT_PASSWORD is not set." }
    if (-not $TimestampUrl) { $TimestampUrl = Get-CepSigningSetting "OPENCUT_CEP_TSA_URL" }
    if (-not $TimestampUrl) { $TimestampUrl = $script:CepDefaultTimestampUrl }
    return [pscustomobject]@{
        Tool         = Resolve-ZxpSignCmd $ZxpSignCmd
        Certificate  = (Resolve-Path -LiteralPath $Certificate).Path
        Password     = $Password
        TimestampUrl = $TimestampUrl
    }
}

function Assert-CepTreeIsSignable {
    param([Parameter(Mandatory = $true)][string]$Path)
    $root = Get-Item -LiteralPath $Path -Force
    $items = @($root) + @(Get-ChildItem -LiteralPath $Path -Recurse -Force)
    $links = @($items | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint })
    if ($links.Count -gt 0) {
        throw ("The CEP panel contains links, which Adobe refuses to load from a signed package: " +
            (($links | ForEach-Object { $_.FullName }) -join ", "))
    }
    $forbidden = @($items | Where-Object { $script:CepForbiddenNames -contains $_.Name })
    if ($forbidden.Count -gt 0) {
        throw ("The staged CEP panel has files that must not be signed into a release: " +
            (($forbidden | ForEach-Object { $_.FullName }) -join ", "))
    }
}

function Invoke-ZxpSignCmd {
    param(
        [Parameter(Mandatory = $true)][string]$Tool,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [string]$Secret = ""
    )
    # Native stderr turns into terminating errors under Stop. Collect it instead.
    $ErrorActionPreference = "Continue"
    $output = (& $Tool @Arguments 2>&1 | ForEach-Object { "$_" }) -join [Environment]::NewLine
    $exitCode = $LASTEXITCODE
    if ($Secret) { $output = $output.Replace($Secret, "<password>") }
    return [pscustomobject]@{ ExitCode = $exitCode; Output = $output.Trim() }
}

function Assert-CepPanelSignature {
    <#
    .SYNOPSIS
        Fail unless Path (a panel folder or a .zxp) carries a valid signature
        that covers every file in it, and, with -RequireTimestamp, a valid
        timestamp so the signature outlives the certificate.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [switch]$RequireTimestamp,
        [string]$ZxpSignCmd = ""
    )
    $tool = Resolve-ZxpSignCmd $ZxpSignCmd
    $result = Invoke-ZxpSignCmd -Tool $tool -Arguments @("-verify", $Path, "-certInfo", "-skipOnlineRevocationChecks")
    if ($result.ExitCode -ne 0 -or $result.Output -notmatch "Signature verified successfully") {
        throw "CEP signature check failed for ${Path} (exit $($result.ExitCode)): $($result.Output)"
    }
    # "Valid timestamp, but outside certificate validity dates" also starts with Valid.
    if ($RequireTimestamp -and $result.Output -notmatch "(?m)^Timestamp:\s*Valid and within certificate validity dates") {
        throw "CEP signature for ${Path} has no valid timestamp: $($result.Output)"
    }
    return $result.Output
}

function Invoke-CepPanelSigning {
    <#
    .SYNOPSIS
        Sign ExtensionDir into ZxpPath, then replace ExtensionDir with the
        signed package's exact contents so META-INF\signatures.xml and
        mimetype ship with the panel.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$ExtensionDir,
        [Parameter(Mandatory = $true)][string]$ZxpPath,
        [string]$Certificate = "",
        [string]$Password = "",
        [string]$TimestampUrl = "",
        [string]$ZxpSignCmd = "",
        # Tests only. A release signature must carry a timestamp.
        [switch]$SkipTimestamp
    )
    $inputs = Resolve-CepSigningInputs -Certificate $Certificate -Password $Password `
        -TimestampUrl $TimestampUrl -ZxpSignCmd $ZxpSignCmd
    $ExtensionDir = (Resolve-Path -LiteralPath $ExtensionDir).Path
    Assert-CepTreeIsSignable $ExtensionDir

    $zxpParent = Split-Path -Parent $ZxpPath
    if ($zxpParent) { New-Item -ItemType Directory -Force -Path $zxpParent | Out-Null }
    if (Test-Path -LiteralPath $ZxpPath) { Remove-Item -LiteralPath $ZxpPath -Force }

    $arguments = @("-sign", $ExtensionDir, $ZxpPath, $inputs.Certificate, $inputs.Password)
    if (-not $SkipTimestamp) { $arguments += @("-tsa", $inputs.TimestampUrl) }
    $signed = Invoke-ZxpSignCmd -Tool $inputs.Tool -Arguments $arguments -Secret $inputs.Password
    if ($signed.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $ZxpPath -PathType Leaf)) {
        throw "ZXPSignCmd -sign failed (exit $($signed.ExitCode)): $($signed.Output)"
    }

    $signedDir = "$ExtensionDir.signed"
    if (Test-Path -LiteralPath $signedDir) { Remove-Item -LiteralPath $signedDir -Recurse -Force }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory($ZxpPath, $signedDir)
    Remove-Item -LiteralPath $ExtensionDir -Recurse -Force
    Move-Item -LiteralPath $signedDir -Destination $ExtensionDir

    $wantTimestamp = -not $SkipTimestamp
    Assert-CepPanelSignature -Path $ZxpPath -RequireTimestamp:$wantTimestamp -ZxpSignCmd $inputs.Tool | Out-Null
    return Assert-CepPanelSignature -Path $ExtensionDir -RequireTimestamp:$wantTimestamp -ZxpSignCmd $inputs.Tool
}
