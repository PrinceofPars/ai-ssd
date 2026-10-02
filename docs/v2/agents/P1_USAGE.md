# P1 Resource Usage and Session Log

Agent: Person 1 (P1)  
Role: Real LLM / KV / Attention / Top-k  
Worktree: `/home/ubuntu/ai-ssd-p1`  
Branch: `v2/p1-real-llm-kv`  
Tmux Session: `p1`  

---

# Current Session

Session:
SESSION-P1-PHASE1-DELIVERY

Started:
2026-10-02T22:15:00+05:30

Last Refresh:
2026-10-03T00:26:00+05:30

Current State:
PHASE 1 COMPLETE — REAL LLM ENGINE, BLOCKIZATION, TRACE GENERATOR & TOP-K EVALUATION DELIVERED

---

# Cumulative Summary

Sessions:
2

Total Duration:
~2 hours 15 minutes

Completed Milestones:
- M0: Reconnaissance & Environment Audit
- M1: Real LLM Model Selection, Physical KV Blockization, Real Access Trace, Top-k Sparse Evaluation, Native C Kernel Build & Testing

Artifacts Created:
- `person1_kv_engine/real_llm/engine.py`
- `person1_kv_engine/real_llm/block_adapter.py`
- `person1_kv_engine/real_llm/trace_generator.py`
- `person1_kv_engine/real_llm/topk_evaluator.py`
- `person1_kv_engine/real_llm/native_benchmark.py`
- `person1_kv_engine/real_llm/run_p1_pipeline.py`
- `person1_kv_engine/real_llm/run_long_eval.py`
- `person1_kv_engine/c_kernel/instorage_attention.so`
- `person1_kv_engine/tests/test_p1_real_llm.py`
- `docs/v2/research/REAL_LLM_KV_RESEARCH.md`
- `docs/v2/proposals/PROP-001-KV-BLOCK-PAGE-GEOMETRY.md`
- `docs/v2/proposals/PROP-002-KV-ACCESS-TRACE-SCHEMA.md`
- `/opt/ai-ssd-v2/traces/real_llm/trace_qwen2.5_0.5b_context512.jsonl` (4.14 MiB, 7872 events)
- `/opt/ai-ssd-v2/traces/real_llm/trace_qwen2.5_0.5b_context512.manifest.json`
- `/opt/ai-ssd-v2/traces/real_llm/trace_qwen2.5_0.5b_context512.sha256`
- `/opt/ai-ssd-v2/results/p1_real_llm_experiment.json`
- `/opt/ai-ssd-v2/results/native_kernel_benchmark.json`

---

# Resource Profiles & Measured Observables

## 1. CPU Usage Observations
- Machine: 8 vCPUs (Intel Xeon Platinum 8488C, 4 cores with 2 threads/core).
- P1 Pinned Core Budget: `torch.set_num_threads(4)`.
- Average CPU load during model inference: ~380% - 400% CPU (exactly 4 cores active).
- P2, P3, OS, and FEMU headroom: 4 full vCPUs remain completely unutilized by P1.
- Generation Speed: **~20.87 tokens/sec** on CPU.
- Prefill Latency: **1.37s** for 687 prompt tokens (~501 tokens/s prefill throughput).

## 2. RAM Footprint
- Total System Memory: 61 GiB (~55 GiB free before execution).
- Model Weight Footprint: **~1.95 GiB** (FP32 precision for `Qwen/Qwen2.5-0.5B`).
- Sliced KV Cache Memory: **16.50 MiB** for 703 tokens across 24 layers (2112 blocks).
- Peak Python Process RSS: **2.32 GiB** (only **3.8% of available machine RAM**).
- Free Memory Remaining: **>58.5 GiB** completely available for FEMU/QEMU VM and P2/P3.

## 3. Model Size & Specifications
- Model: `Qwen/Qwen2.5-0.5B`
- Parameters: 494,032,704 (494M)
- Layers: 24
- Attention Heads ($Q$): 14
- KV Heads ($KV$): 2 (GQA with ratio 7:1)
- Hidden Dimension: 896
- Head Dimension: 64
- Precision: Float32 (CPU)
- Download Cache Size on Disk: ~1.0 GB (`~/.cache/huggingface/hub/`)

## 4. Disk Usage
- Worktree Changes: ~85 KiB code.
- Shared Traces:
  * Short trace: 1.97 MiB (`trace_qwen2.5_0.5b_eval.jsonl`)
  * Long-context trace: 4.14 MiB (`trace_qwen2.5_0.5b_context512.jsonl`)
- Shared Results: < 100 KiB (`p1_real_llm_experiment.json`, `native_kernel_benchmark.json`)
- Free Disk Space Remaining: **188 GiB** available on `/dev/root`.

## 5. Runtimes of Expensive Operations
- GCC Shared Library Compilation: **0.18s**
- Full Unit Test Suite (34 tests): **2.32s**
- 512-Token Context Inference Run: **2.21s**
- In-Storage Top-k Evaluation (5 sparsity budgets): **0.045s**
- Native C Kernel Benchmark (50 iterations x 128 blocks): **0.062s**

---

# Verification Commands for Next Sessions

```bash
# Verify environment in tmux p1
echo "TMUX=$TMUX"
tmux display-message -p '#S'
pwd

# Verify git baseline
git status
git branch --show-current
git rev-parse --short HEAD

# Run full test suite
pytest person1_kv_engine/tests/

# Re-run full pipeline
python person1_kv_engine/real_llm/run_long_eval.py
```
