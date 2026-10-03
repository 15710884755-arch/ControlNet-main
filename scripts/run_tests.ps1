$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Push-Location $repo
try {
    python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    python -m compileall -q config.py cldm ldm
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
