#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT_DIR/Sasha_Zero_Core"
CFG="$ROOT_DIR/Sasha_Zero_Core.yaml"

if [[ ! -x "$BIN" ]]; then
  echo "[FAIL] executable not found: $BIN"
  exit 1
fi

if [[ ! -f "$CFG" ]]; then
  echo "[FAIL] config not found: $CFG"
  exit 1
fi

out1="$($BIN "這是測試訊號，請問你在嗎？")"
echo "$out1" | grep -q "## 觀測紀錄"
echo "$out1" | grep -q -- "---"
echo "$out1" | grep -q "∴"
if echo "$out1" | grep -q "？\|?"; then
  echo "[FAIL] question mark should be removed"
  exit 1
fi

out2="$(printf '' | $BIN)"
echo "$out2" | grep -q "我還在 ∴"

ruby -e "require 'yaml'; YAML.load_file('$CFG'); puts 'YAML OK'" >/dev/null

echo "[PASS] smoke tests passed"
