"""Test bounded decode-time KV flushing during long generation."""
import torch
import pytest
from person1_kv_engine.real_llm.engine import RealLLMEngine
from person1_kv_engine.real_llm.aissd_inference import (
    create_default_storage_backend,
    run_baseline_decode,
    run_aissd_decode,
)


def test_long_decode_bounded_kv_flush():
    engine = RealLLMEngine(model_name="Qwen/Qwen2.5-0.5B", device="cpu", dtype="float32", num_threads=4)
    # Prompt with ~40 tokens ensures sink_tokens (4) and recent_tokens (16) are fully initialized,
    # and recent_k starts at 16 tokens.
    prompt = (
        "The PCI Express standard defines high-speed serial computer expansion bus technology "
        "for NVMe solid state storage controllers and high-throughput accelerator devices in modern enterprise servers."
    )
    input_ids = engine.tokenizer(prompt, return_tensors="pt")["input_ids"]

    # 18 decode tokens: starts with 16 recent tokens, after 16 decode steps reaches 32 >= 32,
    # triggering decode-time block offload into candidate blocks and crossing the 16-token block boundary.
    decode_tokens = 18

    b_res = run_baseline_decode(
        engine.model, engine.tokenizer, input_ids,
        decode_tokens=decode_tokens, seed=42, show_progress=False
    )

    backend = create_default_storage_backend(
        channels=8,
        num_layers=engine.num_layers,
        num_heads=engine.num_kv_heads,
        head_dim=engine.head_dim,
        dtype="float32",
        storage_mode="file",
        enable_batching=True,
    )
    a_res = run_aissd_decode(
        engine.model, engine.tokenizer, input_ids,
        decode_tokens=decode_tokens, top_k_pct=100.0, storage_backend=backend,
        enable_prefetch=True, enable_computational_storage=True, seed=42, show_progress=False
    )

    print("actual input tokens:", input_ids.shape[1])
    b_toks = b_res["token_ids"]
    a_toks = a_res["token_ids"]
    for i in range(len(b_toks)):
        b_str = engine.tokenizer.decode([b_toks[i]])
        a_str = engine.tokenizer.decode([a_toks[i]])
        m_str = "MATCH" if b_toks[i] == a_toks[i] else f"DIFF ({b_toks[i]} != {a_toks[i]})"
        print(f"Step {i:2d}: {m_str:<22} | baseline: {repr(b_str):<15} | aissd: {repr(a_str):<15}")

    # 1. Exact token parity must be preserved across the 16-token block boundary
    assert a_res["token_ids"] == b_res["token_ids"], f"Token mismatch: {a_res['token_ids']} vs {b_res['token_ids']}"

    # 2. Candidate K to host must be zero under computational storage
    assert a_res["candidate_k_bytes_to_host"] == 0

    # 3. Telemetry must show write bytes and read bytes
    assert a_res["total_write_mb"] > 0.0, f"Expected write MB > 0, got {a_res.get('total_write_mb')}"
    assert a_res["total_read_mb"] > 0.0, f"Expected read MB > 0, got {a_res.get('total_read_mb')}"

    # 4. Storage backend must have received completed historical blocks
    storage = getattr(backend, "storage_backend", backend)
    assert storage.bytes_written > 0, "Storage backend must reflect flushed block bytes"
    assert storage.bytes_read > 0, "Storage backend must reflect retrieved block bytes"


if __name__ == "__main__":
    test_long_decode_bounded_kv_flush()
    print("ALL CHECKS PASSED")
