# Standard assertion tests compatible with pytest and unittest
from common.schemas.kv_block import KVBlock, StorageTier
from common.schemas.request import KVRequest, KVOperation
from common.schemas.result import KVResponse, OperationStatus
from common.schemas.metrics import SystemMetrics, MemoryMetrics, LatencyMetrics, StorageMetrics, PrefetchMetrics, FTLMetrics
from common.constants import DEFAULT_BLOCK_SIZE_BYTES


def test_kv_block_defaults():
    block = KVBlock.create_default(
        block_id=1,
        layer_id=0,
        token_start=0,
        token_count=16,
        kv_head_start=0,
        kv_head_count=1,
        head_dim=128,
        dtype="FP16",
    )
    assert block.key_size_bytes == 2048
    assert block.value_size_bytes == 2048
    assert block.total_size_bytes == DEFAULT_BLOCK_SIZE_BYTES
    assert block.total_size_bytes == 4096

    # Test dictionary serialization round-trip
    d = block.to_dict()
    restored = KVBlock.from_dict(d)
    assert restored.block_id == 1
    assert restored.total_size_bytes == 4096


def test_kv_block_fp8():
    block = KVBlock.create_default(
        block_id=2,
        layer_id=1,
        token_start=16,
        token_count=16,
        kv_head_start=0,
        kv_head_count=1,
        head_dim=128,
        dtype="FP8",
    )
    assert block.key_size_bytes == 1024
    assert block.value_size_bytes == 1024
    assert block.total_size_bytes == 2048


def test_request_response_serialization():
    req = KVRequest(
        request_id="req_001",
        operation=KVOperation.KV_TOPK,
        layer_id=12,
        head_id=4,
        candidate_blocks=[10, 11, 12],
        top_k=2,
    )
    d_req = req.to_dict()
    assert d_req["operation"] == "KV_TOPK"
    restored_req = KVRequest.from_dict(d_req)
    assert restored_req.operation == KVOperation.KV_TOPK

    resp = KVResponse(
        request_id="req_001",
        status=OperationStatus.SUCCESS,
        selected_blocks=[10, 12],
        bytes_requested=8192,
        bytes_transferred=4096,
        latency_us=150.0,
        cache_hit=False,
    )
    d_resp = resp.to_dict()
    assert d_resp["status"] == "SUCCESS"
    restored_resp = KVResponse.from_dict(d_resp)
    assert restored_resp.selected_blocks == [10, 12]


def test_system_metrics_structure():
    metrics = SystemMetrics(
        memory=MemoryMetrics(baseline_mb=8192.0, proposed_mb=2048.0, reduction_percent=75.0),
        latency=LatencyMetrics(baseline_ms=100.0, proposed_ms=115.0, overhead_percent=15.0),
        storage=StorageMetrics(bytes_requested=1000, bytes_transferred=200, traffic_reduction_percent=80.0),
        prefetch=PrefetchMetrics(prediction_accuracy=0.88, cache_hit_rate=0.82),
        ftl=FTLMetrics(baseline_read_us=200.0, tensor_aware_read_us=70.0, speedup_x=2.85),
    )
    d = metrics.to_dict()
    restored = SystemMetrics.from_dict(d)
    assert restored.memory.reduction_percent == 75.0
    assert restored.ftl.speedup_x == 2.85
