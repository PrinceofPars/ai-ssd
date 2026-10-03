# AI-SSD V2 Engineering Note: Optimization B ? Key Page Reuse

**Date**: October 3, 2026  
**Author**: Person 1 (Live Inference Optimization Owner)  
**Target File**: person1_kv_engine/real_llm/aissd_inference.py  
**Status**: VERIFIED & BENCHMARKED  
**Branch**: 2-real-llm-kvssd  

---

## 1. What Was Duplicated

During every layer evaluation of the autoregressive decode loop (\text{ layers} \times 15\text{ decode steps} = 360\text{ layer evaluations}$):
1. **Candidate Scoring Step**: The controller scanned all 31 candidate blocks for layer $ by issuing ackend.read_key_page(l_idx, bid) for each candidate ( \text{ calls/layer}$).
2. **Top-$ Winning Fetch Step**: Once winning block IDs were identified (=4$ winning blocks at 10% Top-$), the code re-issued:
   `python
   for _, bid, actual_tokens in top_bids:
       v_blk = self.backend.read_value_page(l_idx, bid)
       k_blk = self.backend.read_key_page(l_idx, bid)  # <-- DUPLICATE READ
   `
This caused \text{ blocks} \times 24\text{ layers} \times 15\text{ decode steps} = \mathbf{1,440}$ redundant flash page reads (,898,240\text{ bytes} / 5.625\text{ MB}$) through the P3 staging buffer and P2 multi-channel FTL.

---

## 2. How Reuse Works

Instead of discarding the Key pages after candidate scanning, select_and_fetch_active_kv() maintains a local reference mapping loaded_k_pages[bid] = k_blk during the candidate scan.
During the subsequent winning block retrieval step, the Key page is fetched from loaded_k_pages[bid] in host memory, issuing **ONLY** ead_value_page(l_idx, bid) to storage:

`
[Candidate Scan]  -->  Read 31 Key pages into local dict loaded_k_pages
         ?
[In-Storage Top-k] -->  AVX2 C kernel selects winning block IDs
         ?
[Winning Fetch]    -->  Read 4 Value pages from storage
                   -->  REUSE 4 Key pages directly from loaded_k_pages (0 storage I/O)
`

The dictionary loaded_k_pages is allocated strictly on the stack within select_and_fetch_active_kv() and is garbage-collected immediately upon layer exit. No extra persistent DRAM is retained.

---

## 3. What Changed in the Implementation

1. **person1_kv_engine/real_llm/aissd_inference.py**:
   - Updated select_and_fetch_active_kv() to record loaded_k_pages[bid] = k_blk.
   - In Step 2 (winning blocks retrieval), replaced self.backend.read_key_page(l_idx, bid) with loaded_k_pages[bid].
2. **person1_kv_engine/tests/test_real_inference.py**:
   - Added 	est_key_page_reuse_correctness() verifying exact request count drop ( \rightarrow 31$ requests per layer) and full memory offload preservation.
   - Added 	est_old_path_k_equals_reused_k() verifying exact bit-for-bit tensor equality (	orch.equal(old_k, reused_k) == True).
3. **person1_kv_engine/tests/test_p2_integration.py**:
   - Updated assertions in 	est_aissd_kv_manager_with_p2_backend() from 35 Key reads down to 31 Key reads.

---

## 4. Correctness Verification

* **Tensor Identity**: 	orch.equal(reused_k, old_path_k) is True.
* **Value Identity**: 	orch.equal(reused_v, old_path_v) is True.
* **Token Output**: Exactly identical generated text and token IDs ([6437, 1584, 6541, 6461, 916, 279, 90690, 24458, 7823, 6541, 916, 279, 90690, 24458, 7823, 15073], .2\%$ match vs baseline).
* **Unit Tests**: /131$ passing (\%$).

---

## 5. Measured Performance Impact

Evaluated using enchmarks/live_inference/run_live_benchmark.py (Qwen2.5-0.5B, CPU 4 threads, Context 512, Decode 16, Seed 42, 3 repetitions):

| Metric | Before Opt B (Post-Opt A) | After Opt B (Key Reuse) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Decode Throughput** | .53 \pm 0.14\text{ tok/s}$ | **.66 \pm 0.09\text{ tok/s}$** | **$+0.13\text{ tok/s}$** |
| **Mean Wall Time** | .9682\text{ s}$ | **.9602\text{ s}$** | **$-8.0\text{ ms}$** |
| **etch_winning_pages Latency** | .44\text{ ms}$ | **.26\text{ ms}$** | **$-33.6\%$ ($-4.18\text{ ms}$)** |
| **Total Storage Requests** | ,324$ | **,884$** | **$-1,440\text{ requests}$ ($-10.1\%$)** |
| **Total Bytes Read** | ,507,840\text{ B}$ (.84\text{ MB}$) | **,609,600\text{ B}$ (.22\text{ MB}$)** | **$-5,898,240\text{ B}$ ($-5.625\text{ MB}$)** |
| **DRAM KV Cache Offload** | .1\%$ | **.1\%$** | Preserved |
| **Speculative Prefetch Accuracy**| \%$ (/284$ useful) | **\%$ (/284$ useful)** | Preserved |
