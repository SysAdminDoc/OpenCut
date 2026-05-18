<#
.SYNOPSIS
    Authenticode-sign Windows release artifacts.
.DESCRIPTION
    F203 signing helper for GitHub Actions. If the signing certificate
    secrets are absent the script exits successfully and leaves artifacts
    unsigned, matching the macOS notarization pattern where operational
    credential setup is separate from repository wiring.
#>

param(
    [string[]]$Paths = @(
        "installer/dist/wpf/OpenCut-WPF-Setup-*.exe",
        "installer/dist/OpenCut-Setup-*.exe"
    ),
    [string]$TimestampUrl = $(if ($env:WINDOWS_CODESIGN_TIMESTAMP_URL) { $env:WINDOWS_CODESIGN_TIMESTAMP_URL } else { "http://timestamp.digicert.com" }),
    [int]$RenewalWarningDays = 90,
    [switch]$FailOnExpiringCert
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-SignTool {
    $cmd = Get-Command signtool.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    $kitsRoot = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\bin"
    if (Test-Path $kitsRoot) {
        $candidate = Get-ChildItem -Path $kitsRoot -Recurse -Filter signtool.exe -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match "\\x64\\signtool\.exe$" } |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($candidate) { return $candidate.FullName }
    }

    throw "signtool.exe not found. Install the Windows SDK on the runner."
}

function Resolve-Artifacts([string[]]$Patterns) {
    $artifacts = @()
    foreach ($pattern in $Patterns) {
        $matches = Get-ChildItem -Path $pattern -File -ErrorAction SilentlyContinue
        foreach ($match in $matches) {
            $artifacts += $match.FullName
        }
    }
    return $artifacts | Sort-Object -Unique
}

function Test-CertRenewalWindow {
    if (-not $env:WINDOWS_CODESIGN_CERT_EXPIRES_AT) {
        Write-Warning "WINDOWS_CODESIGN_CERT_EXPIRES_AT not set; renewal warning check skipped."
        return
    }

    $expiresAt = [DateTimeOffset]::Parse($env:WINDOWS_CODESIGN_CERT_EXPIRES_AT)
    $daysLeft = [math]::Floor(($expiresAt.UtcDateTime - [DateTime]::UtcNow).TotalDays)
    if ($daysLeft -lt $RenewalWarningDays) {
        $message = "Windows code-signing certificate expires in $daysLeft days; renew before release."
        if ($FailOnExpiringCert) { throw $message }
        Write-Warning $message
    } else {
        Write-Host "[codesign] certificate renewal window OK ($daysLeft days left)"
    }
}

if (-not $env:WINDOWS_CODESIGN_PFX_BASE64) {
    Write-Warning "WINDOWS_CODESIGN_PFX_BASE64 is not set; skipping Authenticode signing."
    exit 0
}
if (-not $env:WINDOWS_CODESIGN_PFX_PASSWORD) {
    throw "WINDOWS_CODESIGN_PFX_PASSWORD is required when WINDOWS_CODESIGN_PFX_BASE64 is set."
}

Test-CertRenewalWindow
$artifacts = Resolve-Artifacts $Paths
if (-not $artifacts -or $artifacts.Count -eq 0) {
    throw "No Windows artifacts matched: $($Paths -join ', ')"
}

$signtool = Resolve-SignTool
$pfxPath = Join-Path ([System.IO.Path]::GetTempPath()) "opencut-codesign-$PID.pfx"

try {
    [System.IO.File]::WriteAllBytes(
        $pfxPath,
        [Convert]::FromBase64String($env:WINDOWS_CODESIGN_PFX_BASE64)
    )

    foreach ($artifact in $artifacts) {
        Write-Host "[codesign] signing $artifact"
        & $signtool sign /f $pfxPath /p $env:WINDOWS_CODESIGN_PFX_PASSWORD /fd SHA256 /tr $TimestampUrl /td SHA256 $artifact
        if ($LASTEXITCODE -ne 0) {
            throw "signtool sign failed for $artifact with exit code $LASTEXITCODE"
        }

        & $signtool verify /pa /v $artifact
        if ($LASTEXITCODE -ne 0) {
            throw "signtool verify failed for $artifact with exit code $LASTEXITCODE"
        }
    }
} finally {
    Remove-Item -LiteralPath $pfxPath -Force -ErrorAction SilentlyContinue
}

Write-Host "[codesign] signed $($artifacts.Count) Windows artifact(s)"
