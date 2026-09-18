$ErrorActionPreference = "Stop"

# Run this script from the repository root.
$repo = (Get-Location).Path
if (!(Test-Path (Join-Path $repo "requirements.txt")) -or !(Test-Path (Join-Path $repo "scripts\train.py"))) {
    throw "Run this script from the repository root: the folder must contain requirements.txt and scripts\train.py."
}

$python = Get-Command py -ErrorAction SilentlyContinue
if ($null -eq $python) {
    throw "Python Launcher not found. Install Python 3.12+ from https://www.python.org/downloads/windows/"
}

py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe scripts\train.py
& .\.venv\Scripts\python.exe scripts\run_analysis.py
& .\.venv\Scripts\python.exe -m pytest -q
Write-Host "Setup and verification completed. Activate with: .\.venv\Scripts\Activate.ps1"
