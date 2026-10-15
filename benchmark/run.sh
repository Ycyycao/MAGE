#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: bash run.sh <dataset|all> [base_url] [model]"
    echo "Datasets: webshop websearch math phys travelplan"
    echo "Default base_url: http://localhost:8004/v1"
    echo "Default model: Qwen/Qwen3.6-27B (auto = the one model served at base_url)"
    echo "Environment (all optional):"
    echo "  MAGE_PYTHON   python to run with; default: the conda env 'mage', else .venv/bin/python of this checkout"
    echo "  RUN_NAME      run directory name under logs/<dataset>/; default: a timestamp. Rerunning with the same"
    echo "                name resumes it (tasks already in results.csv are skipped)"
    echo "  WORKERS       concurrent tasks (default 8)"
    echo "  VLLM_EMBEDDING_URL, MAGE_JAVA_HOME   websearch: embedding endpoint, JDK home (default: the python env)"
    echo "  EXTRA_ARGS    more evaluate.py flags, e.g. \"--presence_penalty 1.5\""
}

if [[ $# -lt 1 || $# -gt 3 ]]; then
    usage
    exit 1
fi
case "$1" in
    -h|--help) usage; exit 0 ;;
    all) DATASETS=(webshop websearch math phys travelplan) ;;
    webshop|websearch|math|phys|travelplan) DATASETS=("$1") ;;
    *) usage; exit 1 ;;
esac

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -n "${MAGE_PYTHON:-}" ]]; then
    PY="$MAGE_PYTHON"
elif command -v conda >/dev/null && [[ -x "$(conda info --base)/envs/mage/bin/python" ]]; then
    PY="$(conda info --base)/envs/mage/bin/python"
else
    PY="$WORKDIR/../.venv/bin/python"
fi
if [[ ! -x "$PY" ]]; then
    echo "Python executable not found: $PY (set MAGE_PYTHON)" >&2
    exit 1
fi
ENV_DIR="$(cd "$(dirname "$PY")/.." && pwd)"

BASE_URL="${2:-http://localhost:8004/v1}"
LLM_MODEL="${3:-Qwen/Qwen3.6-27B}"
if [[ "$LLM_MODEL" == auto ]]; then
    LLM_MODEL=$("$PY" -c "import json, sys, urllib.request; print(json.load(urllib.request.urlopen(sys.argv[1].rstrip('/') + '/models'))['data'][0]['id'])" "$BASE_URL")
fi
VLLM_EMBEDDING_URL="${VLLM_EMBEDDING_URL:-http://localhost:8001/v1/embeddings}"
JAVA_HOME="${MAGE_JAVA_HOME:-$ENV_DIR}"
WORKERS="${WORKERS:-8}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
MODEL_TAG="${LLM_MODEL##*/}"
MODEL_TAG="${MODEL_TAG//./_}"
MODEL_TAG="${MODEL_TAG//-/_}"

for DATASET in "${DATASETS[@]}"; do
    SESSION="mage_${DATASET}_${MODEL_TAG}"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Session already exists: $SESSION" >&2
        exit 1
    fi
done

STAMP="${RUN_NAME:-$(date +%Y%m%d-%H%M%S)}"

for DATASET in "${DATASETS[@]}"; do
    SESSION="mage_${DATASET}_${MODEL_TAG}"
    LOG_DIR="$WORKDIR/logs/$DATASET/$STAMP"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/run_${DATASET}.log"
    case "$DATASET" in
        webshop) NUM=150; MAX_RETRY_NUM=2; MAX_STEPS=60 ;;
        websearch) NUM=221; MAX_RETRY_NUM=0; MAX_STEPS=30 ;;
        math) NUM=40; MAX_RETRY_NUM=2; MAX_STEPS=30 ;;
        phys) NUM=20; MAX_RETRY_NUM=1; MAX_STEPS=30 ;;
        travelplan) NUM=270; MAX_RETRY_NUM=0; MAX_STEPS=40 ;;
    esac
    tmux new-session -d -s "$SESSION" \
        "set -o pipefail; cd \"$WORKDIR\" && \
        JAVA_HOME=\"$JAVA_HOME\" \
        VLLM_EMBEDDING_URL=\"$VLLM_EMBEDDING_URL\" \
        \"$PY\" -u evaluate.py \
        --dataset $DATASET \
        --llm \"$LLM_MODEL\" \
        --base_url \"$BASE_URL\" \
        --num $NUM --workers $WORKERS --max_retry_num $MAX_RETRY_NUM \
        --max_steps $MAX_STEPS \
        --log_dir \"$LOG_DIR\" $EXTRA_ARGS 2>&1 | tee -a \"$LOG_FILE\"; \
        status=\$?; echo \"Exit code: \$status\"; exit \$status"
    echo "$DATASET: tmux attach -t $SESSION"
    echo "Log: $LOG_FILE"
done