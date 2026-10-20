#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ruby -e '
  require "yaml"
  config = YAML.load_file(ARGV.fetch(0))
  abort "unexpected persistence claim" unless config.dig("init", "persistence") == false
  abort "unexpected monitoring claim" unless config.dig("input_stream", "continuous_monitoring") == false
  abort "unexpected loop claim" unless config.dig("execution_block", "step_3_sustain", "loop") == false
  abort "missing implementation boundary" unless config.fetch("implementation").fetch("not_implemented").include?("semantic_question_detection")
' "$ROOT_DIR/Sasha_Zero_Core.yaml"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# Both spaces and apostrophes must survive shell and Ruby argument handling.
COPY_DIR="$TMP_DIR/Sasha's runtime copy"
OTHER_DIR="$TMP_DIR/another working directory"
mkdir -p "$COPY_DIR/scripts" "$OTHER_DIR"
cp "$ROOT_DIR/Sasha_Zero_Core" "$ROOT_DIR/Sasha_Zero_Core.yaml" "$COPY_DIR/"
cp "$ROOT_DIR/scripts/smoke_test.sh" "$COPY_DIR/scripts/"
chmod +x "$COPY_DIR/Sasha_Zero_Core"

# An unrelated CWD config must not replace the one beside the executable.
ruby -e '
  require "yaml"
  config = YAML.load_file(ARGV.fetch(0))
  config.fetch("execution_block").fetch("step_2_reconstruction").fetch("structure")["header"] = "CUSTOM HEADER"
  File.write(ARGV.fetch(1), YAML.dump(config))
' "$COPY_DIR/Sasha_Zero_Core.yaml" "$OTHER_DIR/Sasha_Zero_Core.yaml"

expected=$'## 觀測紀錄\n---\n測試\n∴'
actual="$(cd "$OTHER_DIR" && "$COPY_DIR/Sasha_Zero_Core" '測試？')"
[[ "$actual" == "$expected" ]] || { echo '[FAIL] default config depends on CWD'; exit 1; }

# Relative -c paths remain relative to the caller's working directory.
expected_override=$'CUSTOM HEADER\n---\n測試\n∴'
actual="$(cd "$OTHER_DIR" && "$COPY_DIR/Sasha_Zero_Core" -c ./Sasha_Zero_Core.yaml '測試？')"
[[ "$actual" == "$expected_override" ]] || { echo '[FAIL] explicit config override ignored'; exit 1; }

actual="$(cd "$OTHER_DIR" && printf '測試？' | "$COPY_DIR/Sasha_Zero_Core")"
[[ "$actual" == "$expected" ]] || { echo '[FAIL] stdin path handling'; exit 1; }

# Exercise the actual smoke script from outside its repository.
(cd "$OTHER_DIR" && bash "$COPY_DIR/scripts/smoke_test.sh")
echo '[PASS] path regression tests passed'
