param(
    [Parameter(Mandatory = $true)]
    [string]$VideoPath,

    [int]$Fps = 10
)

$ErrorActionPreference = "Stop"

if (!(Test-Path -LiteralPath $VideoPath)) {
    throw "Video not found: $VideoPath"
}

Write-Host "Running pipeline for: $VideoPath"
uv run main.py --video_path $VideoPath --fps $Fps
