param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("audit:check", "audit:esbuild", "build:verify")]
    [string] $Gate,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ForwardArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptByGate = @{
    "audit:check" = "check-advisories.mjs"
    "audit:esbuild" = "check-esbuild-pin.mjs"
    "build:verify" = "verify-build.mjs"
}

$panelScripts = $PSScriptRoot
$scriptPath = Join-Path -Path $panelScripts -ChildPath $scriptByGate[$Gate]

if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) {
    Write-Error "Panel gate script not found: $scriptPath"
    exit 1
}

$node = Get-Command node -ErrorAction Stop
& $node.Source $scriptPath @ForwardArgs
exit $LASTEXITCODE
