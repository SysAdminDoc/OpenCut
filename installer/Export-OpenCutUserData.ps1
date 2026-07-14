[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$UserDataPath,

    [Parameter(Mandatory = $true)]
    [string]$BackupDirectory,

    [Parameter(Mandatory = $true)]
    [string]$ResultFile,

    [switch]$RemoveAfterBackup
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

function Get-NormalizedDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw 'A non-empty directory path is required.'
    }

    $fullPath = [IO.Path]::GetFullPath($Path)
    $root = [IO.Path]::GetPathRoot($fullPath)
    $separators = [char[]]@([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
    if ($fullPath.TrimEnd($separators) -eq $root.TrimEnd($separators)) {
        return $root
    }
    return $fullPath.TrimEnd($separators)
}

function Test-IsSameOrChild {
    param(
        [Parameter(Mandatory = $true)][string]$Candidate,
        [Parameter(Mandatory = $true)][string]$Root
    )

    if ($Candidate.Equals($Root, [StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }
    $prefix = $Root + [IO.Path]::DirectorySeparatorChar
    return $Candidate.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
}

function Assert-SafeUserDataPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    $root = Get-NormalizedDirectory ([IO.Path]::GetPathRoot($Path))
    if ($Path.Equals($root, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'A filesystem root cannot be used as the OpenCut data path.'
    }

    $profile = [Environment]::GetFolderPath([Environment+SpecialFolder]::UserProfile)
    if (-not [string]::IsNullOrWhiteSpace($profile)) {
        $normalizedProfile = Get-NormalizedDirectory $profile
        if ($Path.Equals($normalizedProfile, [StringComparison]::OrdinalIgnoreCase)) {
            throw 'The user profile root cannot be used as the OpenCut data path.'
        }
    }
}

$source = Get-NormalizedDirectory $UserDataPath
$backupRoot = Get-NormalizedDirectory $BackupDirectory
Assert-SafeUserDataPath $source

if (Test-IsSameOrChild $backupRoot $source) {
    throw 'The user-data backup directory must be outside the directory being removed.'
}

if (-not (Test-Path -LiteralPath $source -PathType Container)) {
    Set-Content -LiteralPath $ResultFile -Value 'NOT_FOUND' -Encoding UTF8
    exit 0
}

New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null
$stamp = [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss')
$token = [Guid]::NewGuid().ToString('N').Substring(0, 8)
$backupPath = Join-Path $backupRoot "OpenCut-user-data-$stamp-$token.zip"
$partialPath = "$backupPath.partial"

try {
    $entries = @(Get-ChildItem -LiteralPath $source -Force -Recurse -ErrorAction Stop)
    $reparsePoint = $entries | Where-Object {
        ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
    } | Select-Object -First 1
    if ($null -ne $reparsePoint) {
        throw "User-data backup refuses reparse points: $($reparsePoint.FullName)"
    }

    $sourceFiles = @($entries | Where-Object { -not $_.PSIsContainer })
    foreach ($file in $sourceFiles) {
        $handle = [IO.File]::Open(
            $file.FullName,
            [IO.FileMode]::Open,
            [IO.FileAccess]::Read,
            [IO.FileShare]::None)
        $handle.Dispose()
    }

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::CreateFromDirectory(
        $source,
        $partialPath,
        [IO.Compression.CompressionLevel]::Optimal,
        $false)

    $archive = [IO.Compression.ZipFile]::OpenRead($partialPath)
    try {
        $archiveFiles = @($archive.Entries | Where-Object { -not [string]::IsNullOrEmpty($_.Name) })
        if ($archiveFiles.Count -ne $sourceFiles.Count) {
            throw "Backup validation failed: expected $($sourceFiles.Count) files, found $($archiveFiles.Count)."
        }

        $sourcePrefix = $source + [IO.Path]::DirectorySeparatorChar
        foreach ($file in $sourceFiles) {
            $relativePath = $file.FullName.Substring($sourcePrefix.Length).Replace('\', '/')
            $archiveEntry = $archiveFiles | Where-Object {
                $_.FullName.Replace('\', '/').Equals($relativePath, [StringComparison]::OrdinalIgnoreCase)
            } | Select-Object -First 1
            if ($null -eq $archiveEntry -or $archiveEntry.Length -ne $file.Length) {
                throw "Backup validation failed for $relativePath."
            }

            $entryStream = $archiveEntry.Open()
            try {
                $entryStream.CopyTo([IO.Stream]::Null)
            }
            finally {
                $entryStream.Dispose()
            }
        }
    }
    finally {
        $archive.Dispose()
    }

    Move-Item -LiteralPath $partialPath -Destination $backupPath -ErrorAction Stop

    if ($RemoveAfterBackup) {
        $parent = Split-Path -Parent $source
        $leaf = Split-Path -Leaf $source
        $removalPath = Join-Path $parent ".$leaf.opencut-removing-$([Guid]::NewGuid().ToString('N'))"
        Move-Item -LiteralPath $source -Destination $removalPath -ErrorAction Stop
        try {
            Remove-Item -LiteralPath $removalPath -Recurse -Force -ErrorAction Stop
        }
        catch {
            if (-not (Test-Path -LiteralPath $source) -and (Test-Path -LiteralPath $removalPath)) {
                Move-Item -LiteralPath $removalPath -Destination $source -ErrorAction Stop
            }
            throw
        }
    }

    Set-Content -LiteralPath $ResultFile -Value $backupPath -Encoding UTF8
}
catch {
    if (Test-Path -LiteralPath $partialPath) {
        Remove-Item -LiteralPath $partialPath -Force -ErrorAction SilentlyContinue
    }
    Write-Error $_
    exit 1
}
