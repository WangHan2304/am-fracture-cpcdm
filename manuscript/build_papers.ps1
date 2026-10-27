#Requires -Version 5.0
<#
.SYNOPSIS
    Offline two-pass pdflatex build for the manuscript PDFs.

.DESCRIPTION
    Builds the authoritative EFM submission PDFs (main + supplementary) and, with
    -Ijms, the frozen IJMS pair; with -Sagej, the frozen IJDM/sagej variants.
    Every document is compiled twice so that
    cross-references, \linenumbers and hyperref anchors settle.

    MiKTeX is run with -disable-installer: the machine is offline and on-demand
    package installation would otherwise hang waiting on a package that cannot be
    fetched. All required classes and shims are local (see latex-requirements.md),
    so no download is needed. Non-zero pdflatex exit is treated as a hard failure
    (-halt-on-error already stops on the first error).

    The supplementary documents resolve their figures via ../../simulation/figures
    and the main documents via ../simulation/figures, so each document is compiled
    from its own directory.

.PARAMETER Ijms
    Also build submission_IJMS/manuscript_ijms.tex and its supplementary (the
    frozen IJMS variant, kept as a historical archive; not the reference build).

.PARAMETER Sagej
    Also build manuscript/manuscript_sagej.tex and manuscript/manuscript_main.tex
    (the frozen IJDM variants; not the reference build).

.PARAMETER SkipSi
    Build only the main document(s), skipping every supplementary and the
    attachment PDFs (Highlights, cover letter).

.PARAMETER SkipAtt
    Skip the attachment PDFs (highlights_efm.tex, cover_letter_efm.tex) only.

.PARAMETER Pdflatex
    Path to the pdflatex executable (default: whatever is on PATH).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File manuscript/build_papers.ps1
    powershell -ExecutionPolicy Bypass -File manuscript/build_papers.ps1 -Ijms
    powershell -ExecutionPolicy Bypass -File manuscript/build_papers.ps1 -Sagej
#>
[CmdletBinding()]
param(
    [switch]$Ijms,
    [switch]$Sagej,
    [switch]$SkipSi,
    [switch]$SkipAtt,
    [string]$Pdflatex = 'pdflatex'
)

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path   # .../manuscript
$RepoRoot  = Split-Path -Parent $ScriptDir                     # repo root
$EfmDir    = Join-Path $RepoRoot 'submission_EFM'
$EfmSiDir  = Join-Path $EfmDir 'supplementary'
$IjmsDir   = Join-Path $RepoRoot 'submission_IJMS'
$IjmsSiDir = Join-Path $IjmsDir 'supplementary'

function Build-Pdf {
    param(
        [string]$Dir,
        [string]$Tex
    )
    $texPath = Join-Path $Dir $Tex
    if (-not (Test-Path $texPath)) {
        throw "source not found: $texPath"
    }
    $pdfName  = [IO.Path]::ChangeExtension($Tex, '.pdf')
    $logName  = [IO.Path]::ChangeExtension($Tex, '.log')
    Push-Location $Dir
    try {
        for ($pass = 1; $pass -le 2; $pass++) {
            # MiKTeX prints a benign "check for updates" notice on stderr; running the
            # native call under Stop would turn that into a terminating error, so relax
            # the preference for the invocation, drop both streams to files, and gate on
            # the real exit code only.
            $prev = $ErrorActionPreference
            $ErrorActionPreference = 'Continue'
            & $Pdflatex -interaction=nonstopmode -halt-on-error -disable-installer $Tex 1> "$logName.pass$pass.out" 2> "$logName.pass$pass.err"
            $code = $LASTEXITCODE
            $ErrorActionPreference = $prev
            Remove-Item "$logName.pass$pass.out", "$logName.pass$pass.err" -ErrorAction SilentlyContinue
            if ($code -ne 0) {
                throw "pdflatex EXIT=$code on pass $pass for $Tex (see $(Split-Path $logName -Leaf))"
            }
        }
        $pdfPath = Join-Path $Dir $pdfName
        $pages = 'unknown'
        if (Test-Path $logName) {
            $m = Select-String -Path $logName -Pattern 'Output written on .+\((\d+)\s+page' | Select-Object -Last 1
            if ($m) { $pages = $m.Matches[0].Groups[1].Value }
            $undef = Select-String -Path $logName -Pattern 'undefined references|Citation .* undefined|Reference .* undefined'
            if ($undef) {
                Write-Warning ("{0}: {1} undefined-reference line(s) in log" -f $Tex, $undef.Count)
            }
        }
        if (-not (Test-Path $pdfPath)) {
            throw "pdflatex reported success but $pdfName was not produced"
        }
        Write-Host ("[OK] {0} -> {1}  ({2} pages, {3:N0} bytes)" -f $Tex, $pdfName, $pages, (Get-Item $pdfPath).Length)
    }
    finally {
        Pop-Location
    }
}

Write-Host "== building EFM main (authoritative) =="
Build-Pdf -Dir $EfmDir -Tex 'manuscript_efm.tex'

if (-not $SkipSi) {
    Write-Host "== building EFM supplementary (authoritative) =="
    Build-Pdf -Dir $EfmSiDir -Tex 'supplementary_efm.tex'

    # The Highlights sheet and the cover letter carry the manuscript title, so
    # they are built here as well: a retitled paper must not leave an attachment
    # PDF showing the previous title.
    if (-not $SkipAtt) {
        Write-Host "== building EFM attachment PDFs (Highlights, cover letter) =="
        Build-Pdf -Dir $EfmDir -Tex 'highlights_efm.tex'
        Build-Pdf -Dir $EfmDir -Tex 'cover_letter_efm.tex'
    }
}

if ($Ijms) {
    Write-Host "== building frozen IJMS variant =="
    Build-Pdf -Dir $IjmsDir -Tex 'manuscript_ijms.tex'
    if (-not $SkipSi) {
        Build-Pdf -Dir $IjmsSiDir -Tex 'supplementary_ijms.tex'
    }
}

if ($Sagej) {
    Write-Host "== building frozen IJDM variants (sagej + main) =="
    Build-Pdf -Dir $ScriptDir -Tex 'manuscript_sagej.tex'
    Build-Pdf -Dir $ScriptDir -Tex 'manuscript_main.tex'
}

Write-Host "`nbuild_papers: all requested PDFs compiled with EXIT=0."
