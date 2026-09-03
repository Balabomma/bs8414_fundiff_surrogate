<#
    Launch this project's BS 8414 Part1 Streamlit app.

    .\run_app.ps1              # http://localhost:8501
    .\run_app.ps1 -Port 8502   # a second app alongside the first

    SHARED FILE — byte-identical in bs8414_KAN_surrogate, bs8414_MLP_surrogate,
    bs8414_fundiff_surrogate and bs8414_fundiff_kan_surrogate. It picks the app
    this project actually ships (thermocouple or slice), so one copy serves all
    four. Never hand-edit one copy: edit this file and re-copy it.
#>
param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$Project = $PSScriptRoot
Set-Location $Project

# ── the app this project ships ────────────────────────────────────────────
$App = $null
foreach ($candidate in @("app_part1.py", "app_fundiff_part1.py")) {
    if (Test-Path (Join-Path $Project $candidate)) { $App = $candidate; break }
}
if (-not $App) {
    throw "No Part1 Streamlit app in $Project (expected app_part1.py or app_fundiff_part1.py)."
}

# ── this project's own venv, never a shared one ───────────────────────────
$Activate = Join-Path $Project "venv\Scripts\Activate.ps1"
if (-not (Test-Path $Activate)) { throw "No venv at $Activate." }
& $Activate

# ── the shipped material table; the app never reads the FDS corpus ────────
if (-not (Test-Path (Join-Path $Project "app_assets\part1_materials.json"))) {
    Write-Host "Material table missing - exporting it from the FDS corpus once..." -ForegroundColor Yellow
    python (Join-Path (Split-Path $Project -Parent) "export_app_assets.py")
}

# ── GPU status, so a busy card is visible before the app claims it ────────
$gpu = & nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>$null
if ($LASTEXITCODE -eq 0 -and $gpu) {
    Write-Host "GPU: $gpu" -ForegroundColor DarkGray
    $used = [int](($gpu -split ",")[1] -replace "[^\d]", "")
    if ($used -gt 18000) {
        Write-Host "  >18 GB already in use - a training run is probably active." -ForegroundColor Yellow
        Write-Host "  The app falls back to CPU only if CUDA is unavailable, not if it is merely full." -ForegroundColor Yellow
    }
} else {
    Write-Host "nvidia-smi unavailable - the app will run on CPU." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Launching $App on http://localhost:$Port" -ForegroundColor Green
Write-Host "  Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

streamlit run $App --server.port $Port --server.headless false --browser.gatherUsageStats false
