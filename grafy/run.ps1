<#
run.ps1 — spouštěcí skript pro Windows (PowerShell)
# usage: .\run.ps1 [command] [file]
# example: .\run.ps1 summary 01.tg
#
# Skript vytvoří venv (pokud neexistuje), aktivuje ho, nainstaluje balíčky z lokální složky 'packages'
# a spustí `graph_tool.py`.
#>
param(
  [string]$Command = "summary",
  [string]$File = "01.tg",
  [string]$Extra = ""
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

if (-not (Test-Path .venv)) {
  python -m venv .venv
}

# Aktivace v aktuálním sezení
. .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

$packages = Join-Path $scriptDir "packages"
if (-not (Test-Path $packages)) {
  Write-Host "Složka s balíčky 'packages' nenalezena v $packages. Přesvědčte se, že jste zkopírovali offline balíček."
} else {
  python -m pip install --no-index --find-links $packages -r requirements.txt
}

if ($Command -eq "summary") {
  python .\graph_tool.py summary $File $Extra
} else {
  python .\graph_tool.py $Command $File $Extra
}
