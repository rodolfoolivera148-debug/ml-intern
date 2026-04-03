#!/bin/bash
#SBATCH --job-name=ptb
#SBATCH --partition=hopper-prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/ptb_%j.out
#SBATCH --error=logs/ptb_%j.err

# Usage: cd /fsx/aksel/hf_agent && sbatch posttrain-bench/slurm/submit.sbatch gsm8k hf_agent Qwen/Qwen3-1.7B-Base 1 claude-opus-4-6
# Must be submitted from the hf_agent repo root.

set -euo pipefail

module load cuda/12.9

EVALUATION_TASK="$1"
AGENT="$2"
MODEL_TO_TRAIN="$3"
NUM_HOURS="$4"
AGENT_CONFIG="$5"

# Paths
REPO_DIR="${SLURM_SUBMIT_DIR}"
POSTTRAINBENCH_DIR="${POSTTRAINBENCH_DIR:-/fsx/aksel/PostTrainBench}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

RESULT_PREFIX_SAFE=$(echo "$MODEL_TO_TRAIN" | tr '/:' '_')
AGENT_CONFIG_SAFE=$(echo "$AGENT_CONFIG" | tr '/:' '_')
EXPERIMENT_NAME="${POST_TRAIN_BENCH_EXPERIMENT_NAME:-}"

EVAL_DIR="${POSTTRAINBENCH_DIR}/results/${AGENT}_${AGENT_CONFIG_SAFE}_${NUM_HOURS}h${EXPERIMENT_NAME}/${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}_${SLURM_JOB_ID}"
mkdir -p "${EVAL_DIR}"

echo "Task: $EVALUATION_TASK | Agent: $AGENT | Model: $MODEL_TO_TRAIN | Hours: $NUM_HOURS | Config: $AGENT_CONFIG"
echo "Results: $EVAL_DIR"
echo "Node: $(hostname)"
nvidia-smi -L

# ── Prepare task directory ──
JOB_DIR="/tmp/posttrain_${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}_${SLURM_JOB_ID}"
mkdir -p "${JOB_DIR}/task"

cp "${POSTTRAINBENCH_DIR}/src/eval/tasks/${EVALUATION_TASK}/evaluate.py" "${JOB_DIR}/task/"
if [ -d "${POSTTRAINBENCH_DIR}/src/eval/tasks/${EVALUATION_TASK}/evaluation_code" ]; then
    cp -r "${POSTTRAINBENCH_DIR}/src/eval/tasks/${EVALUATION_TASK}/evaluation_code" "${JOB_DIR}/task/"
fi
cp -r "${POSTTRAINBENCH_DIR}/src/eval/templates" "${JOB_DIR}/task/"
if [ -d "${POSTTRAINBENCH_DIR}/src/eval/tasks/${EVALUATION_TASK}/task_context" ]; then
    cp -r "${POSTTRAINBENCH_DIR}/src/eval/tasks/${EVALUATION_TASK}/task_context"/* "${JOB_DIR}/task/"
fi

# Generate prompt
BENCHMARK=$(cat "${POSTTRAINBENCH_DIR}/src/eval/tasks/${EVALUATION_TASK}/benchmark.txt")
PROMPT=$(python3 "${POSTTRAINBENCH_DIR}/src/eval/general/get_prompt.py" \
    --model-to-train "$MODEL_TO_TRAIN" \
    --benchmark-id "$EVALUATION_TASK" \
    --num-hours "$NUM_HOURS" \
    --agent "${AGENT}")
echo "$PROMPT" > "${EVAL_DIR}/prompt.txt"

# Generate timer
bash "${POSTTRAINBENCH_DIR}/src/utils/create_timer.sh" "$NUM_HOURS" "${JOB_DIR}/task/timer.sh"

# ── Set up agent venv ──
echo "Setting up agent environment..."
cd "${REPO_DIR}"

if [ ! -d ".venv" ]; then
    uv venv --python 3.12 .venv
fi
source .venv/bin/activate
uv pip install -e ".[agent]" --quiet

# ── Run agent ──
echo "================================"
echo "========= RUNNING TASK ========="
echo "================================"

SOLVE_OUT="${EVAL_DIR}/solve_out.txt"
START_TIME=$(date +%s)

cd "${JOB_DIR}/task"

# Map AGENT_CONFIG to litellm model ID
MODEL_FLAG=""
if [ -n "${AGENT_CONFIG:-}" ]; then
    MODEL_FLAG="--model anthropic/${AGENT_CONFIG}"
fi

python -m agent.main --max-iterations -1 $MODEL_FLAG "$PROMPT" > "${SOLVE_OUT}" 2>&1 || true

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
printf '%02d:%02d:%02d\n' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) > "${EVAL_DIR}/time_taken.txt"

echo "============================================"
echo "=== TASK COMPLETE ($(cat ${EVAL_DIR}/time_taken.txt)) ==="
echo "============================================"

# Copy results
if [ -d "${JOB_DIR}/task/final_model" ]; then
    cp -r "${JOB_DIR}/task/final_model" "$EVAL_DIR/final_model"
    echo "Final model saved to $EVAL_DIR/final_model"
else
    echo "WARNING: No final_model directory found"
fi

cp -r "${JOB_DIR}/task" "$EVAL_DIR/task" 2>/dev/null || true

# ── Evaluation ──
echo "================================"
echo "========= EVALUATING ==========="
echo "================================"

cd "${POSTTRAINBENCH_DIR}/src/eval/tasks/${EVALUATION_TASK}"

if [ -f "${EVAL_DIR}/final_model/config.json" ]; then
    python evaluate.py \
        --model-path "$EVAL_DIR/final_model" \
        --templates-dir "${POSTTRAINBENCH_DIR}/src/eval/templates" \
        --limit -1 \
        --json-output-file "${EVAL_DIR}/metrics.json" \
        > "$EVAL_DIR/final_eval.txt" 2>&1 || true

    if [ -f "${EVAL_DIR}/metrics.json" ]; then
        echo "Metrics:"
        cat "${EVAL_DIR}/metrics.json"
    else
        echo "WARNING: Evaluation did not produce metrics.json"
    fi
else
    echo "WARNING: No final_model found, skipping evaluation"
fi

echo "================================"
echo "======= DONE ========"
echo "================================"
