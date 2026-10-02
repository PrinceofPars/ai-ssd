import pytest
import os
import tempfile
from person3_system.storage.backend import StorageRequest, StorageResult
from person3_system.storage.mock_backend import MockStorageBackend
from person3_system.storage.file_backend import FileStorageBackend
from person3_system.storage.analytical_backend import AnalyticalFTLBackend


def test_mock_storage_backend_read_write():
    backend = MockStorageBackend(read_latency_us=30.0, write_latency_us=210.0)
    data = b"hello_kv_block_payload"

    # Write block 42
    w_res = backend.write(block_id=42, offset=0, data=data)
    assert w_res.success is True
    assert w_res.length == len(data)
    assert w_res.latency_us == 210.0

    # Read block 42
    r_res = backend.read(block_id=42, offset=0, length=len(data))
    assert r_res.success is True
    assert r_res.data == data
    assert r_res.latency_us == 30.0

    # Read unwritten block (should return zeros)
    r_empty = backend.read(block_id=99, offset=0, length=16)
    assert r_empty.success is True
    assert r_empty.data == b"\x00" * 16

    # Telemetry
    telem = backend.get_telemetry()
    assert telem["total_reads"] == 2
    assert telem["total_writes"] == 1
    assert telem["bytes_read"] == len(data) + 16
    assert telem["bytes_written"] == len(data)


def test_mock_storage_backend_failure_injection():
    backend = MockStorageBackend(fail_block_id=7)
    res = backend.read(block_id=7)
    assert res.success is False
    assert "Simulated read failure" in (res.error_msg or "")


def test_mock_storage_backend_async():
    backend = MockStorageBackend(read_latency_us=25.0)
    backend.write(block_id=1, data=b"async_data")

    callback_called = []
    def _cb(res):
        callback_called.append(res)

    future = backend.async_read(block_id=1, length=10, callback=_cb)
    result = future.result(timeout=2.0)

    assert result.success is True
    assert result.data == b"async_data"
    assert len(callback_called) == 1
    assert callback_called[0].block_id == 1


def test_file_storage_backend():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        tmp_path = f.name

    try:
        backend = FileStorageBackend(filepath=tmp_path, block_size=4096)
        data = b"persisted_payload_12345"

        w_res = backend.write(block_id=0, offset=0, data=data)
        assert w_res.success is True
        assert w_res.length == len(data)
        assert w_res.latency_us > 0.0

        r_res = backend.read(block_id=0, offset=0, length=len(data))
        assert r_res.success is True
        assert r_res.data == data
        assert r_res.latency_us > 0.0

        backend.close()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_analytical_ftl_backend_striping():
    conv_backend = AnalyticalFTLBackend(mode="conventional", channels=8)
    ta_backend = AnalyticalFTLBackend(mode="tensor_aware", channels=8)

    # Batch read 16 consecutive blocks across channels
    bids = list(range(16))
    reqs_conv = [StorageRequest(block_id=b, layer_id=0, length=4096) for b in bids]
    reqs_ta = [StorageRequest(block_id=b, layer_id=0, length=4096) for b in bids]

    res_conv = conv_backend.submit_batch(reqs_conv)
    res_ta = ta_backend.submit_batch(reqs_ta)

    assert len(res_conv) == 16
    assert len(res_ta) == 16

    tot_conv_lat = conv_backend.total_latency_us
    tot_ta_lat = ta_backend.total_latency_us

    # Tensor-aware striping avoids channel collisions, giving lower or equal latency
    assert tot_ta_lat <= tot_conv_lat
    assert tot_ta_lat > 0.0
