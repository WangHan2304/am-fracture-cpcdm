#!/usr/bin/env bash
set -e
OUT="/mnt/d/20260618断裂模型论文/simulation/output"
for m in 316L Ti64 AlSi10Mg; do
  cp "$HOME/damask_${m}_fft32/macro_curve.csv" "$OUT/damask_${m}_macro.csv"
  echo "== $m: $(wc -l < "$OUT/damask_${m}_macro.csv") lines =="
  head -3 "$OUT/damask_${m}_macro.csv"
done
