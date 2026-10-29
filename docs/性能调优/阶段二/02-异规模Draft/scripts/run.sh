#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
export PYTHONPATH=.. PYTHONDONTWRITEBYTECODE=1
PYTHON=/home/ubuntu/miniconda3/envs/ai-nanovllm/bin/python
DIR=docs/性能调优/阶段二/02-异规模Draft
if "$PYTHON" "$DIR/scripts/check_correctness.py" 2>&1 | tee "$DIR/data/check-rebuild.log"; then :; else echo "Rebuild greedy mismatch; benchmark is diagnostic only."; fi
if "$PYTHON" "$DIR/scripts/check_correctness.py" --cache 2>&1 | tee "$DIR/data/check-cache.log"; then :; else echo "Cache greedy mismatch; benchmark is diagnostic only."; fi
for item in normal:1 rebuild:1 cache:1 cache:2 rebuild:2; do
  mode="${item%:*}"; run="${item#*:}"
  "$PYTHON" "$DIR/scripts/compare.py" --mode "$mode" --repeats 2 --output "$DIR/data/$mode-$run.json" > "$DIR/data/$mode-$run.log" 2>&1
done
"$PYTHON" "$DIR/scripts/trace_logits.py" > "$DIR/data/trace-logits.json"
