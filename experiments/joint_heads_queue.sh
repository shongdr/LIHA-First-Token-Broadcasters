#!/usr/bin/env bash
# Runs joint_heads.py for the ten 1-7B models, one queue per GPU (joint_heads_plan.md). From the repo root:
#   GPUS="0 1 2 3" bash experiments/joint_heads_queue.sh   # default: four GPUs
#   GPUS="0 1" bash experiments/joint_heads_queue.sh
#   EXTRA=1 bash experiments/joint_heads_queue.sh          # also OLMo-3-7B and SmolLM3-3B (not counted in the paper)
# Needs the environment of requirements.txt and a Hugging Face login with access to Llama 3.2 and Gemma 3.
# Re-running is safe: finished conditions are skipped. Logs: logs/joint_<model>.log; outputs: out/<model>-joint.
set -u
GPUS=${GPUS:-"0 1 2 3"}
read -r -a G <<< "$GPUS"
N=${#G[@]}
# largest first, dealt in snake order so each GPU gets a mix of sizes
MODELS=(qwen2.5-7b-instruct qwen3-4b-instruct gemma3-4b-instruct qwen2.5-3b-instruct llama3.2-3b-instruct
        qwen3-1.7b-instruct qwen-instruct gemma3-1b-instruct llama3.2-1b-instruct olmo2-1b-instruct)
if [ "${EXTRA:-0}" = "1" ]; then MODELS+=(olmo3-7b-instruct smollm3-instruct); fi
mkdir -p logs

CUDA_VISIBLE_DEVICES=${G[0]} python experiments/joint_heads.py selftest --model qwen3-1.7b-instruct \
    > logs/joint_selftest.log 2>&1 || { echo "selftest failed, see logs/joint_selftest.log"; exit 1; }
grep -c PASS logs/joint_selftest.log

declare -a Q
for i in "${!MODELS[@]}"; do
  r=$(( i / N )); c=$(( i % N )); [ $(( r % 2 )) -eq 1 ] && c=$(( N - 1 - c ))
  Q[$c]="${Q[$c]:-} ${MODELS[$i]}"
done
for c in "${!G[@]}"; do
  (
    for m in ${Q[$c]:-}; do
      echo "$(date '+%F %T') GPU ${G[$c]} start $m"
      CUDA_VISIBLE_DEVICES=${G[$c]} python experiments/joint_heads.py run --model "$m" >> "logs/joint_$m.log" 2>&1 \
        && python experiments/joint_heads.py report --model "$m" >> "logs/joint_$m.log" 2>&1
      echo "$(date '+%F %T') GPU ${G[$c]} done  $m (exit $?)"
    done
  ) &
done
wait
echo "summaries: out/<model>-joint/summary.md"
