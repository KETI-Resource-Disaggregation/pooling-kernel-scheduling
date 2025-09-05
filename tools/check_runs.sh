#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-archive/logs}"
echo "[scan] $ROOT"
tot=$(find "$ROOT" -name metrics.json | wc -l)
echo "[found] metrics.json = $tot"

echo
echo "[missing] 모드별 metrics.json 누락 케이스:"
# rep-* / MODE 폴더는 있는데 metrics.json이 없는 경우
while IFS= read -r d; do
  for M in UNBOUND SALUS BLESSISH OURS; do
    p="$d/$M/metrics.json"
    [[ -d "$d/$M" && ! -f "$p" ]] && echo " - $d/$M"
  done
done < <(find "$ROOT" -type d -name 'rep-*' | sort)
