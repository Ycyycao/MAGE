#!/bin/bash
# Set up MAGE with uv instead of conda: a .venv in this checkout, then the evaluation data.
#   bash scripts/setup_uv.sh [core|all]
#   core: agentcore + math/phys/travelplan dependencies (no pyserini/faiss/datasets; enough without websearch)
#   all:  everything in requirements.txt (pip install -e .), as the conda instructions do
# Python: MAGE_PYTHON_VERSION (default 3.12). Downloads go through HF_ENDPOINT when it is set (e.g. a mirror).
# run.sh picks this .venv up when there is no conda env "mage" (or set MAGE_PYTHON).
set -euo pipefail
cd "$(dirname "$0")/.."
WHAT="${1:-core}"
[ -x .venv/bin/python ] || uv venv --python "${MAGE_PYTHON_VERSION:-3.12}" .venv
PIP=(uv pip install --python .venv/bin/python)  # explicit: a UV_PYTHON in the environment would win over VIRTUAL_ENV
case "$WHAT" in
    core)
        "${PIP[@]}" --no-deps -e .
        grep -E '^(openai|requests|pydantic|sympy|antlr4|beautifulsoup4)' requirements.txt | sed 's/ //g' \
            | xargs "${PIP[@]}" ;;
    all) "${PIP[@]}" -e . ;;
    *) echo "usage: $0 [core|all]" >&2; exit 2 ;;
esac
bash benchmark/download_data.sh
.venv/bin/python -c "import agentcore, openai, sympy; print('agentcore ready:', '$PWD/.venv')"
