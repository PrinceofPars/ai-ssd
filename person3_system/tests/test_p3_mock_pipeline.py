# Standard assertion tests compatible with pytest and unittest
from person1_kv_engine.mock_ssd import MockSSD
from person2_ssd.mock_kv_engine import MockKVEngine
from person3_system.api.ai_ssd import AISSDSystem
from person3_system.api.requests import RequestFactory
from person3_system.prefetch.prefetcher import SpeculativePrefetcher
from person3_system.integration.orchestrator import SystemOrchestrator
from person3_system.integration.pipeline import InferencePipeline
from common.schemas.result import OperationStatus


def test_api_with_mocks():
    mock_kv = MockKVEngine(layers=4, heads=8)
    mock_ssd = MockSSD()
    prefetcher = SpeculativePrefetcher()

    api = AISSDSystem(kv_engine=mock_kv, ssd_engine=mock_ssd, prefetcher=prefetcher)

    # Test TOPK request
    req = RequestFactory.create_topk_request(
        layer_id=0,
        head_id=0,
        candidate_blocks=[10, 11, 12, 13, 14],
        top_k=2,
    )
    resp = api.execute(req)
    assert resp.status == OperationStatus.SUCCESS
    assert len(resp.selected_blocks) == 2
    assert resp.bytes_transferred == 2 * 4096


def test_pipeline_with_mocks():
    mock_kv = MockKVEngine(layers=4, heads=8)
    mock_ssd = MockSSD()
    orchestrator = SystemOrchestrator(kv_engine=mock_kv, ssd_engine=mock_ssd, enable_prefetch=True)
    pipeline = InferencePipeline(orchestrator=orchestrator)

    result = pipeline.run_decode_layer(
        layer_id=1,
        head_id=2,
        candidate_blocks=list(range(20)),
        top_k=4,
    )
    assert result["status"] == "SUCCESS"
    assert len(result["selected_blocks"]) == 4
