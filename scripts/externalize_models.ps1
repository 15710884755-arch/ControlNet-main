[CmdletBinding()]
param(
    [string]$Destination = 'D:\ControlNet-models',
    [switch]$Apply
)

$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$source = Join-Path $repo 'models'

if (-not (Test-Path -LiteralPath $source -PathType Container)) {
    throw "Model directory not found: $source"
}

$sourceResolved = (Resolve-Path -LiteralPath $source).Path
if (-not $sourceResolved.StartsWith($repo, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to move outside repository: $sourceResolved"
}

if (Test-Path -LiteralPath $Destination) {
    throw "Destination already exists: $Destination"
}

$files = Get-ChildItem -LiteralPath $sourceResolved -Recurse -File
$sizeGb = [math]::Round((($files | Measure-Object -Property Length -Sum).Sum) / 1GB, 2)

Write-Output "Repository : $repo"
Write-Output "Source     : $sourceResolved ($($files.Count) files, $sizeGb GB)"
Write-Output "Destination: $Destination"

if (-not $Apply) {
    Write-Output 'Dry run only. Re-run with -Apply to move the directory and create a junction.'
    return
}

Move-Item -LiteralPath $sourceResolved -Destination $Destination
New-Item -ItemType Junction -Path $sourceResolved -Target $Destination | Out-Null

$probe = Join-Path $sourceResolved 'control_sd15_canny.pth'
if (-not (Test-Path -LiteralPath $probe)) {
    throw "Junction verification failed: $probe"
}

Write-Output "Moved models to $Destination and created junction at $sourceResolved"
