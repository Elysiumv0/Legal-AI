# setup_python312.ps1 - cai Python 3.12 song song (khong dung 3.15)
#
# Ly do: Python 3.15 qua moi → khong co wheel cho underthesea_core (Rust),
# PyYAML (C), va hau nhu toan bo requirements.txt. May cung khong co MSVC.
# 3.12 la ban moi nhat co wheel rong rai cho torch/transformers/qdrant/faiss.
#
# Cai vao user scope, KHONG dong vao PATH he thong; repo dung venv riêng.

$ErrorActionPreference = "Stop"

function Step($n, $msg) { Write-Host "`n=== [$n] $msg ===" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "  OK: $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "  WARN: $msg" -ForegroundColor Yellow }

$Version = "3.12.8"
$Base = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312"
$Venv = "C:\Users\ADMIN\Legal-AI\.venv"

# -- 0. da co chua? -------------------------------------------------------------
Step 0 "Check existing install"
$py = Join-Path $Base "python.exe"
if (Test-Path $py) {
    Ok "Python 3.12 da co tai $Base"
} else {
    # -- 1. tai installer user-scope (khong can admin) -------------------------
    Step 1 "Download Python $Version installer"
    $tmp = Join-Path $env:TEMP "py312"
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    $exe = Join-Path $tmp "python-$Version-amd64.exe"
    $url = "https://www.python.org/ftp/python/$Version/python-$Version-amd64.exe"
    Write-Host "  $url"
    Invoke-WebRequest $url -OutFile $exe
    Ok "downloaded $(Split-Path $exe -Leaf)"

    # -- 2. cai user-scope, KHONG them vao PATH he thong ------------------------
    #    InstallAllUsers=0 (user), PrependPath=0 (khong dung PATH toan cuc)
    Step 2 "Install (user scope, no system PATH change)"
    $args = @(
        "/quiet",
        "InstallAllUsers=0",
        "TargetDir=$Base",
        "PrependPath=0",
        "Include_launcher=0",
        "Include_pip=1",
        "Include_test=0",
        "AssociateFiles=0",
        "Shortcuts=0"
    )
    Start-Process -FilePath $exe -ArgumentList $args -Wait -NoNewWindow
    if (-not (Test-Path $py)) {
        throw "Install xong nhung khong thay $py"
    }
    Ok "installed: $py"
}

& $py --version

# -- 3. tao venv cho repo -------------------------------------------------------
Step 3 "Create venv"
if (Test-Path $Venv) {
    Warn "venv da ton tai tai $Venv - giu nguyen, khong xoa"
} else {
    & $py -m venv $Venv
    Ok "venv created: $Venv"
}

$vpy = Join-Path $Venv "Scripts\python.exe"
& $vpy --version
& $vpy -c "import sys; print('base_prefix:', sys.base_prefix)"
& $vpy -m pip install --quiet --upgrade pip setuptools wheel
Ok "pip upgraded in venv"

# -- 4. cai dependencies --------------------------------------------------------
Step 4 "Install requirements.txt (wheel-only, no source builds)"
# --only-binary :avoid=:None → chi lay wheel co san, khong build tu source
# (chinh la thu da lam pip mat 10 phut compile Rust/MSVC tren 3.15)
& $vpy -m pip install --only-binary :all: -r "C:\Users\ADMIN\Legal-AI\requirements.txt"
Ok "requirements installed"

# -- 5. verify import quan trong -------------------------------------------------
Step 5 "Verify imports"
& $vpy -c @'
import importlib
mods = ["underthesea", "rank_bm25", "yaml", "numpy", "torch", "transformers",
        "qdrant_client", "pydantic", "pdfplumber", "fastapi", "gradio"]
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  OK   {m}")
    except Exception as e:
        print(f"  FAIL {m}: {type(e).__name__}: {e}")
'@

Write-Host "`nDONE. Tu nay dung: $Venv\Scripts\python.exe" -ForegroundColor Green
Write-Host "Hoac chay .venv\Scripts\activate truoc khi lam viec voi repo." -ForegroundColor Green
