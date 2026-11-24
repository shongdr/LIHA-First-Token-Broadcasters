#!/usr/bin/env bash
# Runs format_grid.py for every pair on one GPU, then copies the compact results to results/format-grid
# (format_grid_plan.md). From the repo root:
#   GPU=0 bash experiments/format_grid_queue.sh
#   PAIRS="qwen3-1.7b gemma3-1b" GPU=0 bash experiments/format_grid_queue.sh   # a subset
# One 72 GB GPU holds every pair (the largest are Qwen2.5-7B in bf16 and Gemma-3-4B in fp32). Needs the environment
# of requirements.txt and a Hugging Face login with access to Gemma 3. Re-running is safe: finished conditions are
# skipped. Logs: logs/format_<pair>.log.
set -u
GPU=${GPU:-0}
PAIRS=${PAIRS:-"qwen2.5-1.5b qwen3-1.7b gemma3-1b olmo2-1b qwen2.5-3b gemma3-4b qwen2.5-7b"}
mkdir -p logs results/format-grid
for p in $PAIRS; do
  echo "$(date '+%F %T') start $p"
  if CUDA_VISIBLE_DEVICES=$GPU python experiments/format_grid.py run --pair "$p" >> "logs/format_$p.log" 2>&1; then
    mkdir -p "results/format-grid/$p"
    cp "out/format-grid/$p/meta.json" "out/format-grid/$p/labels.json" "results/format-grid/$p/"
    gzip -c "out/format-grid/$p/gens.jsonl" > "results/format-grid/$p/gens.jsonl.gz"
    echo "$(date '+%F %T') done  $p"
  else
    echo "$(date '+%F %T') FAILED $p, see logs/format_$p.log"
  fi
done
python experiments/format_grid.py report --out results/format-grid
