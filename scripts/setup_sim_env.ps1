<#
Create or update the repository-local simulator environment with uv.

The `.python-version` pin keeps this project on CPython 3.11: the system
Python may be 3.14, which is outside the wheel support range of some Newton
dependencies.  uv downloads the pinned interpreter when needed and creates
the gitignored repository-local `.venv` from `pyproject.toml` + `uv.lock`.
#>
[CmdletBinding()]
param(
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install it with: irm https://astral.sh/uv/install.ps1 | iex"
}

Push-Location $Repo
try {
    $SyncArgs = @("sync", "--all-groups")
    if ($Python) { $SyncArgs += @("--python", $Python) }
    & uv @SyncArgs
} finally {
    Pop-Location
}
$VenvPython = Join-Path $Repo ".venv\Scripts\python.exe"
& $VenvPython -c "import mujoco, newton, warp; print('Ready:', 'mujoco', mujoco.__version__, '| newton', newton.__version__)"
