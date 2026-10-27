# setup_ocr.ps1 - install OCR stack for Legal-AI on Windows
#   poppler            : pdf2image backend, renders PDF to images
#   tesseract          : OCR engine
#   vie.traineddata    : Vietnamese language pack for tesseract
#   pdf2image/pytesseract (pip)
#
# Safe to re-run: each step skips what is already installed.
# NOTE: all ASCII on purpose - Windows PowerShell 5.1 reads .ps1 as ANSI,
# so non-ASCII in this file would garble on your machine.

$ErrorActionPreference = "Stop"

function Step($n, $msg) { Write-Host "`n=== [$n] $msg ===" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "  OK: $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "  WARN: $msg" -ForegroundColor Yellow }

# -- 0. pip packages -----------------------------------------------------------
Step 0 "pip: pdf2image, pytesseract"
python -m pip install --quiet --upgrade pdf2image pytesseract
Ok "pdf2image + pytesseract installed"

# -- 1. poppler (backend for pdf2image) ---------------------------------------
$PopplerBin = $null
$existing = Get-Command pdftoppm -ErrorAction SilentlyContinue
if ($existing) {
    $PopplerBin = Split-Path $existing.Source
    Ok "poppler already on PATH: $PopplerBin"
} else {
    Step 1 "poppler: download Windows build"
    $Candidates = @(
        (Join-Path $env:LOCALAPPDATA "poppler\Library\bin"),
        (Join-Path $env:LOCALAPPDATA "poppler\bin"),
        "C:\poppler\Library\bin",
        "C:\poppler\bin"
    )
    foreach ($c in $Candidates) {
        if (Test-Path (Join-Path $c "pdftoppm.exe")) { $PopplerBin = $c; break }
    }
    if (-not $PopplerBin) {
        $tmp = Join-Path $env:TEMP "poppler-dl"
        New-Item -ItemType Directory -Force -Path $tmp | Out-Null
        # GitHub API rate-limits by IP, so try known release URLs directly.
        # Newest first; stop at the first one that answers HTTP 200.
        $Urls = @(
            "https://github.com/oschwartz10612/poppler-windows/releases/download/v25.7.0-0/Release-25.7.0-0.zip",
            "https://github.com/oschwartz10612/poppler-windows/releases/download/v25.5.0-1/Release-25.5.0-1.zip",
            "https://github.com/oschwartz10612/poppler-windows/releases/download/v24.08.0-1/Release-24.08.0-1.zip",
            "https://github.com/oschwartz10612/poppler-windows/releases/download/v23.08.0-0/Release-23.08.0-0.zip"
        )
        $zip = $null
        foreach ($u in $Urls) {
            try {
                $resp = Invoke-WebRequest -Uri $u -Method Head -UseBasicParsing -TimeoutSec 30
                if ($resp.StatusCode -eq 200) {
                    $zip = Join-Path $tmp (Split-Path $u -Leaf)
                    Write-Host "  downloading $(Split-Path $u -Leaf) ..."
                    Invoke-WebRequest -Uri $u -OutFile $zip -TimeoutSec 300
                    break
                }
            } catch {
                Write-Host "  unavailable: $u"
            }
        }
        if (-not $zip -or -not (Test-Path $zip)) {
            throw "Could not download poppler from any known release URL. Download manually from https://github.com/oschwartz10612/poppler-windows/releases and unzip to $env:LOCALAPPDATA\poppler"
        }
        # User scope, not Program Files - no admin rights needed and nothing
        # outside your profile is written.
        $dest = Join-Path $env:LOCALAPPDATA "poppler"
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        Expand-Archive -Path $zip -DestinationPath $dest -Force
        Remove-Item $zip -Force
        # pdftoppm.exe may be nested in the archive - scan for it
        $found = Get-ChildItem -Path $dest -Recurse -Filter pdftoppm.exe | Select-Object -First 1
        if (-not $found) { throw "Extracted but pdftoppm.exe not found under $dest" }
        $PopplerBin = $found.DirectoryName
    }
    Ok "poppler bin: $PopplerBin"
}

# -- 2. tesseract --------------------------------------------------------------
$TessExe = $null
$existing = Get-Command tesseract -ErrorAction SilentlyContinue
if ($existing) {
    $TessExe = $existing.Source
    Ok "tesseract already on PATH: $TessExe"
} else {
    Step 2 "tesseract engine"
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "  winget install UB-Mannheim.TesseractOCR (may need admin)..."
        winget install --id UB-Mannheim.TesseractOCR -e --silent `
            --accept-package-agreements --accept-source-agreements
        $cand = @(
            "C:\Program Files\Tesseract-OCR\tesseract.exe",
            "$env:LOCALAPPDATA\Programs\Tesseract-OCR\tesseract.exe"
        )
        foreach ($c in $cand) { if (Test-Path $c) { $TessExe = $c; break } }
    } else {
        Warn "no winget - download tesseract-w64-setup from github UB-Mannheim/tesseract manually"
    }
    if (-not $TessExe) {
        throw "tesseract not installed. This terminal also does not have the new PATH - open a NEW terminal and re-run this script."
    }
    Ok "tesseract: $TessExe"
}

# -- 3. Vietnamese language pack -----------------------------------------------
Step 3 "tessdata: vie"
$TessDir = Split-Path $TessExe
# The installer's own tessdata dir is admin-owned, so we cannot drop files
# there without elevation. Tesseract honours TESSDATA_PREFIX *first*, so we
# put our own copy in the user profile and point the env var at it.
$TessData = Join-Path $env:LOCALAPPDATA "tessdata"
New-Item -ItemType Directory -Force -Path $TessData | Out-Null

$ViePath = Join-Path $TessData "vie.traineddata"
if (-not (Test-Path $ViePath)) {
    # tessdata_best: better accuracy than tessdata_fast, matters for legal text
    Write-Host "  downloading vie.traineddata (tessdata_best)..."
    Invoke-WebRequest "https://github.com/tesseract-ocr/tessdata_best/raw/main/vie.traineddata" -OutFile $ViePath
}
Ok "vie.traineddata: $ViePath ($([math]::Round((Get-Item $ViePath).Length/1MB,1)) MB)"

# -- 4. PATH + TESSDATA_PREFIX (user scope) ------------------------------------
Step 4 "PATH + TESSDATA_PREFIX (user scope)"
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
foreach ($dir in @($PopplerBin, $TessDir)) {
    if ($userPath -notlike "*$dir*") {
        [Environment]::SetEnvironmentVariable("Path", "$userPath;$dir", "User")
        $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
        Ok "added to PATH: $dir"
    } else { Ok "already on PATH: $dir" }
}
[Environment]::SetEnvironmentVariable("TESSDATA_PREFIX", $TessData, "User")
Ok "TESSDATA_PREFIX = $TessData"
Warn "This terminal does NOT see the new PATH yet. Steps below use absolute paths, so they still run."

# -- 5. Verify -----------------------------------------------------------------
Step 5 "Verify"
# poppler prints its version to stderr, which with ErrorActionPreference=Stop
# aborts the script - route through cmd so the exit code stays 0.
$ver = & cmd /c "`"$PopplerBin\pdftoppm.exe`" -v 2>&1" | Select-Object -First 1
Ok "poppler: $ver"
$ver2 = & $TessExe --version 2>&1 | Select-Object -First 1
Ok "tesseract: $ver2"
$langs = & $TessExe --list-langs 2>&1 | Out-String
if ($langs -match 'vie') { Ok "vie language available" } else { Warn "vie NOT listed - check TESSDATA_PREFIX=$TessData" }

# -- 5b. Smoke test: OCR page 1 of the scanned PDF ------------------------------
# The user-scope env var written above is not visible to THIS process either, so
# set both PATH and TESSDATA_PREFIX in-process first.
$env:TESSDATA_PREFIX = $TessData
if ($env:Path -notlike "*$TessDir*") { $env:Path = "$env:Path;$TessDir" }
if ($env:Path -notlike "*$PopplerBin*") { $env:Path = "$env:Path;$PopplerBin" }
# NOTE: target filename contains non-ASCII chars (U+00FF U+00FE). PowerShell 5.1
# reads this .ps1 as ANSI so a literal path would garble - let Python glob it.
Step "5b" "Smoke test OCR on page 1"
$py = @'
import glob, os, sys
from pdf2image import convert_from_path
import pytesseract
src = None
for pat in ("*2.pdf", "*.pdf"):
    hits = sorted(glob.glob(os.path.join(r"C:\Users\ADMIN\Downloads", pat)))
    hits = [h for h in hits if os.path.basename(h).startswith("\u00ff\u00fe")]
    if hits:
        src = hits[0]
        break
if src is None:
    print("SKIP: scanned sample PDF not found in Downloads (name starts with U+00FF U+00FE + '2.pdf')")
    sys.exit(0)
print("file:", os.path.basename(src))
pages = convert_from_path(
    src, dpi=200, first_page=1, last_page=1,
    poppler_path=sys.argv[1],
)
txt = pytesseract.image_to_string(pages[0], lang="vie")
print("chars:", len(txt.strip()))
print("sample:", txt.strip()[:200].replace(chr(10), " | "))
'@
$tmp = Join-Path $env:TEMP "ocr_smoke.py"
Set-Content -Path $tmp -Value $py -Encoding utf8
& python $tmp $PopplerBin
Remove-Item $tmp -Force

Write-Host "`nDONE. If the smoke test shows real Vietnamese words, the OCR stack is ready." -ForegroundColor Green
