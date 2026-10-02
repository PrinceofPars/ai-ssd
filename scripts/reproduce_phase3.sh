#!/usr/bin/env bash
# ==============================================================================
# AI-SSD V2 Phase 3: Real LLM / KV-Cache Evaluation Reproducibility Script
# ==============================================================================
# Role: Person 1 (P1)
# Worktree: ~/ai-ssd-p1
# Branch: v2/p1-real-llm-kv
# Model: Qwen/Qwen2.5-0.5B (CPU, FP32, 4 threads)
# Contexts: 128, 256, 512, 1024, 2048, 4096
# Sparsity Budgets: 1%, 5%, 10%, 20%, 50%
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

echo "======================================================================"
echo "REPRODUCING AI-SSD V2 PHASE 3 EVALUATION"
echo "======================================================================"
echo "Project Root: ${PROJECT_ROOT}"
echo "Git Commit:   $(git rev-parse HEAD)"
echo "Git Branch:   $(git branch --show-current)"
echo "Timestamp:    $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "Host CPU:     $(lscpu | grep 'Model name' | sed 's/Model name:[ \t]*//')"
echo "======================================================================"

# 1. Activate Python virtual environment
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
else
    echo "ERROR: Virtualenv not found at ${PROJECT_ROOT}/.venv"
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TORCH_NUM_THREADS=4

# 2. Compile Native In-Storage C Kernel
echo -e "\n[Step 1/3] Compiling Native In-Storage C Kernel..."
python -c "from person1_kv_engine.c_kernel.compile_kernel import compile_c_kernel; compile_c_kernel()"

# 3. Execute Phase 3 Evaluation Suite
echo -e "\n[Step 2/3] Executing Real LLM Evaluation Suite across 6 context lengths..."
python person1_kv_engine/real_llm/phase3_evaluator.py

# 4. Run PyTest Verification Suite
echo -e "\n[Step 3/3] Running PyTest Verification Suite..."
pytest person1_kv_engine/tests/ -v

echo -e "\n======================================================================"
echo "PHASE 3 EVALUATION REPRODUCTION COMPLETE"
echo "Results stored in: /opt/ai-ssd-v2/results/p1/"
echo "Traces stored in:  /opt/ai-ssd-v2/traces/real_llm/"
echo "======================================================================"
