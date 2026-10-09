# run_nonlocal_matrix.ps1  (ASCII only -- avoids PS 5.1 ANSI source decoding)
# Launch the Ti64 0deg nonlocal/local comparison matrix: one python process per case.
# Usage:
#   powershell -File run_nonlocal_matrix.ps1 -Root <simulation dir>
param(
  [Parameter(Mandatory = $true)][string]$Root,
  [string]$Py = 'C:\Users\ZJ03\miniconda3\python.exe'
)

$ErrorActionPreference = 'Stop'
$script  = Join-Path $Root 'fullfield_3d_cpfem_nonlocal.py'
$logdir  = Join-Path $Root 'output\nonlocal_cases'

if (-not (Test-Path -LiteralPath $script))  { throw "missing script: $script" }
if (-not (Test-Path -LiteralPath $Root))    { throw "missing root: $Root" }
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
if (-not (Test-Path -LiteralPath $logdir))  { throw "cannot create logdir: $logdir" }
Write-Output "root    = $Root"
Write-Output "script  = $script"
Write-Output "logdir  = $logdir"

$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONIOENCODING = 'utf-8'

$cases = @(
  @{ id = 'ng3_local_e1e3';       a = @('--ng','3','--eps-step','0.001','--max-strain','0.20') },
  @{ id = 'ng3_lc0333_e1e3';      a = @('--ng','3','--lc','0.3333333333333333','--eps-step','0.001','--max-strain','0.20') },
  @{ id = 'ng3_lc0250_e1e3';      a = @('--ng','3','--lc','0.25','--eps-step','0.001','--max-strain','0.20') },
  @{ id = 'ng4_local_e1e3';       a = @('--ng','4','--eps-step','0.001','--max-strain','0.20') },
  @{ id = 'ng4_lc0333_e1e3';      a = @('--ng','4','--lc','0.3333333333333333','--eps-step','0.001','--max-strain','0.20') },
  @{ id = 'ng4_lc0250_e1e3';      a = @('--ng','4','--lc','0.25','--eps-step','0.001','--max-strain','0.20') },
  @{ id = 'ng3_local_e2e3_repro'; a = @('--ng','3','--eps-step','0.002','--max-strain','0.17') }
)

$procs = @()
foreach ($c in $cases) {
  $log = Join-Path $logdir ($c.id + '.log')
  $err = Join-Path $logdir ($c.id + '.err')
  $argl = @($script, '--case-id', $c.id) + $c.a
  $p = Start-Process -FilePath $Py -ArgumentList $argl -NoNewWindow -PassThru `
         -RedirectStandardOutput $log -RedirectStandardError $err
  $procs += [pscustomobject]@{ id = $c.id; pid = $p.Id; log = $log }
  Start-Sleep -Milliseconds 500
}
$procs | ForEach-Object { Write-Output ("launched " + $_.id + " pid=" + $_.pid) }
Start-Sleep -Seconds 20
Write-Output '--- after 20 s: alive? ---'
foreach ($p in $procs) {
  $alive = $null -ne (Get-Process -Id $p.pid -ErrorAction SilentlyContinue)
  Write-Output ("  " + $p.id + " alive=" + $alive)
  if (-not $alive) {
    $e = Join-Path $logdir (($p.id) + '.err')
    if (Test-Path -LiteralPath $e) {
      Write-Output ('    ERR: ' + ((Get-Content -LiteralPath $e -Raw -Encoding utf8) -replace "`r?`n", ' | '))
    }
  }
}
Write-Output '--- waiting for all cases (this is the long part) ---'
foreach ($p in $procs) {
  try { Wait-Process -Id $p.pid -ErrorAction Stop } catch { }
  Write-Output ("finished " + $p.id + " pid=" + $p.pid)
}
Write-Output 'ALL_CASES_FINISHED'
Get-ChildItem -LiteralPath $logdir -Filter '*.json' | Select-Object Name, Length |
  Format-Table -AutoSize | Out-String -Width 200 | Write-Output
