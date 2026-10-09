#!/usr/bin/env bash
# Usage: wsl_run_skfem.sh <script.py> [ngrid]
export HOME=/home/zj
source /home/zj/fenix_venv/bin/activate
NG="${2:-4}"
# Select the REAL repo dir (not the mojibake twin): the one whose fullfield_skfem.py
# honours the FF_NGRID env override (only present in the edited tree).
F=""
for c in $(ls -1 /mnt/d/20260618*/simulation/fullfield_skfem.py 2>/dev/null); do
  if grep -q "FF_NGRID" "$c" 2>/dev/null; then F="$c"; break; fi
done
if [ -z "$F" ]; then echo "ERROR: no FF_NGRID-aware fullfield_skfem.py found"; exit 2; fi
SIM=$(dirname "$F")
ln -sfn "$SIM" /home/zj/sim
export SIMHOME=/home/zj/sim
export FF_NGRID="$NG"
echo "SIM=$SIM FF_NGRID=$NG"
cd /home/zj/sim
python "/home/zj/sim/$1"
