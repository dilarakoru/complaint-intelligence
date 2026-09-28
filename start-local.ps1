Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath 'artifacts/model.pkl')) { python train.py; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }
python -m uvicorn app:app --host 127.0.0.1 --port 8001
