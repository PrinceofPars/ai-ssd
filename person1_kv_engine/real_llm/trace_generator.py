"""Real KV Access Trace Generator for AI-SSD V2.

Generates reproducible, hardware-accurate JSONL storage access traces from real LLM inference.
Target output: /opt/ai-ssd-v2/traces/real_llm/
"""

from typing import List, Dict, Tuple, Any, Optional
import json
import os
import hashlib
import time
from datetime import datetime
import numpy as np

from common.schemas.kv_block import KVBlock, StorageTier
from person1_kv_engine.c_kernel.kernel_binding import get_native_c_kernel


class RealKVTraceGenerator:
    """Produces JSONL storage traces and manifest metadata from real model inference."""

    def __init__(
        self,
        output_dir: str = "/opt/ai-ssd-v2/traces/real_llm",
        trace_version: str = "2.0",
        git_commit: str = "db7e0f8",
    ):
        self.output_dir = output_dir
        self.trace_version = trace_version
        self.git_commit = git_commit
        os.makedirs(self.output_dir, exist_ok=True)
        self.kernel = get_native_c_kernel()

    def generate_trace_from_run(
        self,
        inference_result: Dict[str, Any],
        layer_blocks: Dict[int, List[Tuple[KVBlock, Dict[str, np.ndarray]]]],
        workload_name: str = "qwen2.5_0.5b_eval",
        top_k_percent: float = 10.0,
    ) -> Tuple[str, str, str]:
        """Generates JSONL trace, manifest JSON, and SHA-256 checksum from real inference run.
        
        Returns:
            Tuple of (trace_file_path, manifest_file_path, sha256_hash)
        """
        trace_filename = f"trace_{workload_name}.jsonl"
        manifest_filename = f"trace_{workload_name}.manifest.json"
        sha_filename = f"trace_{workload_name}.sha256"

        trace_path = os.path.join(self.output_dir, trace_filename)
        manifest_path = os.path.join(self.output_dir, manifest_filename)
        sha_path = os.path.join(self.output_dir, sha_filename)

        events: List[Dict[str, Any]] = []
        event_id = 0
        base_time_ns = int(time.time() * 1e9)

        meta = inference_result["model_metadata"]
        num_layers = meta["num_layers"]
        num_q_heads = meta["num_attention_heads"]
        num_kv_heads = meta["num_kv_heads"]
        head_dim = meta["head_dim"]
        gqa_ratio = meta["gqa_ratio"]
        prompt_tokens = inference_result["prompt_tokens"]
        gen_tokens = inference_result["generated_tokens"]
        total_steps = len(inference_result["step_queries"])

        # 1. PREFILL PHASE: Write initial KV blocks
        for l_idx in range(num_layers):
            blocks = layer_blocks[l_idx]
            for block_desc, _ in blocks:
                event = {
                    "event_id": event_id,
                    "query_id": 0,
                    "step": 0,
                    "layer_id": l_idx,
                    "head_id": block_desc.kv_head_start,
                    "operation": "PREFILL_WRITE",
                    "block_id": block_desc.block_id,
                    "token_start": block_desc.token_start,
                    "token_end": block_desc.token_start + block_desc.token_count,
                    "sub_page": "BOTH",
                    "byte_size": block_desc.total_size_bytes,
                    "tier": block_desc.storage_tier,
                    "hotness": block_desc.hotness,
                    "candidate_blocks": [],
                    "selected_blocks": [block_desc.block_id],
                    "timestamp_ns": base_time_ns + event_id * 100,
                }
                events.append(event)
                event_id += 1

        # 2. DECODE PHASE: Token-by-token attention retrieval
        for step_idx, step_q in enumerate(inference_result["step_queries"]):
            query_id = step_idx + 1

            for l_idx in range(num_layers):
                q_layer = step_q[l_idx]  # [q_heads, head_dim]
                blocks = layer_blocks[l_idx]

                # Partition blocks into DRAM (sinks + recent) and SSD candidates
                dram_blocks = [b for b, _ in blocks if b.storage_tier == StorageTier.DRAM.value]
                ssd_entries = [(b, payload) for b, payload in blocks if b.storage_tier == StorageTier.SSD.value]

                candidate_ids = [b.block_id for b, _ in ssd_entries]
                k_val = max(1, int(len(candidate_ids) * (top_k_percent / 100.0))) if candidate_ids else 0

                # If SSD candidates exist, simulate in-storage Top-k filtering
                selected_ssd_ids = []
                if ssd_entries and k_val > 0:
                    # Native C kernel or NumPy scoring
                    if self.kernel.is_available():
                        # Transform entries for C kernel: list of (block_id, {"k": ..., "v": ...})
                        ck_entries = [(b.block_id, {"k": payload["k"][0], "v": payload["v"][0]}) for b, payload in ssd_entries]
                        topk_ids, _, _ = self.kernel.compute_topk(q_layer, ck_entries, k_val)
                        selected_ssd_ids = topk_ids.tolist()
                    else:
                        # Reference score
                        scores = []
                        scale = 1.0 / np.sqrt(head_dim)
                        for b, payload in ssd_entries:
                            k_block = payload["k"]  # [kv_heads, tokens, head_dim]
                            dots = np.einsum("hd,ktd->ht", q_layer, k_block) * scale
                            score = float(np.max(dots))
                            scores.append((score, b.block_id))
                        scores.sort(key=lambda x: x[0], reverse=True)
                        selected_ssd_ids = [bid for _, bid in scores[:k_val]]

                # Log DRAM cache hits (fast path)
                for b in dram_blocks:
                    event = {
                        "event_id": event_id,
                        "query_id": query_id,
                        "step": step_idx + 1,
                        "layer_id": l_idx,
                        "head_id": b.kv_head_start,
                        "operation": "DECODE_READ",
                        "block_id": b.block_id,
                        "token_start": b.token_start,
                        "token_end": b.token_start + b.token_count,
                        "sub_page": "BOTH",
                        "byte_size": b.total_size_bytes,
                        "tier": "DRAM",
                        "hotness": b.hotness,
                        "candidate_blocks": [],
                        "selected_blocks": [b.block_id],
                        "timestamp_ns": base_time_ns + event_id * 100,
                    }
                    events.append(event)
                    event_id += 1

                # Log In-Storage Top-k Filter Event (Scanning 4 KiB Key pages in SSD)
                if candidate_ids:
                    filter_bytes = len(candidate_ids) * blocks[0][0].key_size_bytes
                    event = {
                        "event_id": event_id,
                        "query_id": query_id,
                        "step": step_idx + 1,
                        "layer_id": l_idx,
                        "head_id": 0,
                        "operation": "TOPK_FILTER",
                        "block_id": candidate_ids[0],
                        "token_start": 0,
                        "token_end": 0,
                        "sub_page": "KEY",
                        "byte_size": filter_bytes,
                        "tier": "SSD",
                        "hotness": 0.5,
                        "candidate_blocks": candidate_ids,
                        "selected_blocks": selected_ssd_ids,
                        "topk_k": k_val,
                        "timestamp_ns": base_time_ns + event_id * 100,
                    }
                    events.append(event)
                    event_id += 1

                    # Log Host retrieval of selected blocks (Fetching 4 KiB Value pages over PCIe)
                    for sel_id in selected_ssd_ids:
                        target_b = next(b for b, _ in ssd_entries if b.block_id == sel_id)
                        event = {
                            "event_id": event_id,
                            "query_id": query_id,
                            "step": step_idx + 1,
                            "layer_id": l_idx,
                            "head_id": target_b.kv_head_start,
                            "operation": "TOPK_FETCH",
                            "block_id": sel_id,
                            "token_start": target_b.token_start,
                            "token_end": target_b.token_start + target_b.token_count,
                            "sub_page": "VALUE",
                            "byte_size": target_b.value_size_bytes,
                            "tier": "SSD",
                            "hotness": target_b.hotness,
                            "candidate_blocks": candidate_ids,
                            "selected_blocks": selected_ssd_ids,
                            "timestamp_ns": base_time_ns + event_id * 100,
                        }
                        events.append(event)
                        event_id += 1

        # 3. Serialize to JSONL
        with open(trace_path, "w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")

        # 4. Compute SHA-256
        sha256 = hashlib.sha256()
        with open(trace_path, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        digest = sha256.hexdigest()

        with open(sha_path, "w", encoding="utf-8") as f:
            f.write(f"{digest}  {trace_filename}\n")

        # 5. Build Manifest
        manifest = {
            "model_name": meta["model_name"],
            "model_architecture": "Qwen2ForCausalLM",
            "num_layers": num_layers,
            "num_query_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "gqa_ratio": gqa_ratio,
            "dtype": meta["dtype"],
            "tokens_per_block": blocks[0][0].token_count,
            "key_page_bytes": blocks[0][0].key_size_bytes,
            "value_page_bytes": blocks[0][0].value_size_bytes,
            "logical_block_bytes": blocks[0][0].total_size_bytes,
            "workload_name": workload_name,
            "prompt": inference_result["prompt"],
            "prompt_tokens": prompt_tokens,
            "generated_tokens": gen_tokens,
            "total_tokens": prompt_tokens + gen_tokens,
            "total_steps": total_steps,
            "total_events": len(events),
            "top_k_percent": top_k_percent,
            "trace_format_version": self.trace_version,
            "git_commit": self.git_commit,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "sha256_checksum": digest,
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return trace_path, manifest_path, digest
