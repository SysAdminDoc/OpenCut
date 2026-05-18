<#
.SYNOPSIS
Run the WPF installer xUnit contract suite.
#>

[CmdletBinding()]
param(
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$testProject = Join-Path $repoRoot "installer\tests\OpenCut.Installer.Tests\OpenCut.Installer.Tests.csproj"

if (-not (Test-Path -LiteralPath $testProject)) {
    throw "WPF installer test project missing: $testProject"
}

Write-Host "[F212] Running WPF installer xUnit suite"
& dotnet test $testProject -c $Configuration --nologo
if ($LASTEXITCODE -ne 0) {
    throw "WPF installer xUnit suite failed with exit code $LASTEXITCODE"
}

Write-Host "[F212] WPF installer xUnit suite passed"
