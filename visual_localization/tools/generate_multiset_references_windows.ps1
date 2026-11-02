param(
  [Parameter(Mandatory=$true)][string]$Glb,
  [ValidateSet("quick","balanced","dense")][string]$Preset = "balanced",
  [string]$Blender = "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
  [string]$Output = ".\synthetic_refs_random",
  [switch]$DryRun
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path $Blender)) { throw "Blender not found: $Blender" }
$glbPath = (Resolve-Path $Glb).Path
$scriptPath = (Resolve-Path (Join-Path $PSScriptRoot "blender_render_references.py")).Path
$outPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Output))

switch ($Preset) {
  "quick" {
    $spacing = 0.85; $yaw = 45; $maxViews = 900; $fov = 95
    $heights = @(0.45, 1.20, 1.50); $pitches = @(-20, 0, 20)
  }
  "dense" {
    $spacing = 0.50; $yaw = 30; $maxViews = 4000; $fov = 95
    $heights = @(0.45, 1.05, 1.35, 1.55, 1.75); $pitches = @(-30, -15, 0, 15, 30)
  }
  default {
    $spacing = 0.65; $yaw = 30; $maxViews = 2400; $fov = 95
    $heights = @(0.45, 1.10, 1.45, 1.70); $pitches = @(-25, -10, 0, 10, 25)
  }
}

$args = @(
  '-b','-P',$scriptPath,'--',
  '--glb',$glbPath,
  '--output',$outPath,
  '--profile','random-phone',
  '--spacing',"$spacing",
  '--heights'
) + ($heights | ForEach-Object { "$_" }) + @(
  '--primary-height','1.45',
  '--yaw-step',"$yaw",
  '--pitches'
) + ($pitches | ForEach-Object { "$_" }) + @(
  '--fov-deg',"$fov",
  '--max-views',"$maxViews"
)
if ($DryRun) { $args += '--dry-run' }

Write-Host "Generating $Preset random-view references from $glbPath"
Write-Host "  spacing=$spacing m; yaw=$yaw deg; max=$maxViews; FOV=$fov deg"
Write-Host "  heights=$($heights -join ', '); pitches=$($pitches -join ', ')"
& $Blender @args
if ($LASTEXITCODE -ne 0) { throw "Blender reference generation failed with exit code $LASTEXITCODE" }

if ($DryRun) {
  Write-Host "DONE (dry run). Report: $(Join-Path $outPath 'generation_report.json')"
  exit 0
}

$json = Join-Path $outPath 'references.json'
$images = Join-Path $outPath 'images'
$report = Join-Path $outPath 'generation_report.json'
if (-not (Test-Path $json) -or -not (Test-Path $images)) {
  throw "Generator finished but references.json/images are missing."
}
$zip = "$outPath.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
# Only references.json + images are required by the importer. The report remains beside the ZIP.
Compress-Archive -Path $json,$images -DestinationPath $zip -Force
Write-Host "DONE"
Write-Host "Reference folder: $outPath"
Write-Host "Generation report: $report"
Write-Host "Upload ZIP:       $zip"
Write-Host "Upload GLB:       $glbPath"
