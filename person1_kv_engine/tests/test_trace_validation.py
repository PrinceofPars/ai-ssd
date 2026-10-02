"""Unit Test Validating Real KV Access Trace Integrity and Schema Compliance."""

import unittest
import os
import json
import hashlib

from common.schemas.trace import CanonicalTraceRecord, TraceManifest


class TestRealTraceValidation(unittest.TestCase):
    """Validates the actual generated Qwen2.5-0.5B trace against canonical V2 contracts."""

    @classmethod
    def setUpClass(cls):
        cls.trace_dir = "/opt/ai-ssd-v2/traces/real_llm"
        cls.trace_file = os.path.join(cls.trace_dir, "trace_qwen2.5_0.5b_context512.jsonl")
        cls.manifest_file = os.path.join(cls.trace_dir, "trace_qwen2.5_0.5b_context512.manifest.json")
        cls.sha_file = os.path.join(cls.trace_dir, "trace_qwen2.5_0.5b_context512.sha256")

        if not os.path.exists(cls.trace_file):
            raise FileNotFoundError(f"Trace file missing: {cls.trace_file}")
        if not os.path.exists(cls.manifest_file):
            raise FileNotFoundError(f"Manifest missing: {cls.manifest_file}")

    def test_sha256_checksum(self):
        """Validates that trace file SHA-256 matches manifest and .sha256 file."""
        sha256 = hashlib.sha256()
        with open(self.trace_file, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        digest = sha256.hexdigest()

        with open(self.manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(digest, manifest["sha256_checksum"])

        if os.path.exists(self.sha_file):
            with open(self.sha_file, "r", encoding="utf-8") as f:
                sha_content = f.read().strip().split()[0]
            self.assertEqual(digest, sha_content)

    def test_manifest_metadata(self):
        """Validates manifest model parameters and physical dimensions."""
        manifest = TraceManifest.from_file(self.manifest_file)
        self.assertEqual(manifest.model_name, "Qwen/Qwen2.5-0.5B")
        self.assertEqual(manifest.num_layers, 24)
        self.assertEqual(manifest.num_kv_heads, 2)
        self.assertEqual(manifest.head_dim, 64)
        self.assertEqual(manifest.tokens_per_block, 16)
        self.assertEqual(manifest.key_page_bytes, 4096)
        self.assertEqual(manifest.value_page_bytes, 4096)
        self.assertEqual(manifest.logical_block_bytes, 8192)
        self.assertEqual(manifest.total_tokens, 703)
        self.assertEqual(manifest.total_events, 7872)

    def test_all_events_semantic_validity(self):
        """Validates all 7872 events for internal semantic and physical consistency."""
        valid_ops = {"PREFILL_WRITE", "DECODE_READ", "TOPK_FILTER", "TOPK_FETCH"}
        total_events = 0
        prev_eid = -1
        prev_ts = 0

        with open(self.trace_file, "r", encoding="utf-8") as f:
            for line in f:
                total_events += 1
                rec = json.loads(line)

                # Parse into canonical schema
                crec = CanonicalTraceRecord.from_dict(rec)
                self.assertEqual(crec.event_id, prev_eid + 1)
                self.assertGreaterEqual(crec.timestamp_ns, prev_ts)
                prev_eid = crec.event_id
                prev_ts = crec.timestamp_ns

                self.assertIn(crec.operation, valid_ops)
                self.assertIn(crec.head_id, (0, 1))  # 2 KV heads
                self.assertIn(crec.layer_id, range(24))

                if crec.operation == "PREFILL_WRITE":
                    self.assertEqual(crec.step, 0)
                    self.assertEqual(crec.sub_page, "BOTH")
                    self.assertEqual(crec.byte_size, 8192)
                    self.assertTrue(crec.is_write)
                    self.assertFalse(crec.is_read)

                elif crec.operation == "DECODE_READ":
                    self.assertGreaterEqual(crec.step, 1)
                    self.assertEqual(crec.tier, "DRAM")
                    self.assertEqual(crec.sub_page, "BOTH")
                    self.assertEqual(crec.byte_size, 8192)
                    self.assertTrue(crec.is_read)
                    self.assertFalse(crec.is_write)

                elif crec.operation == "TOPK_FILTER":
                    self.assertGreaterEqual(crec.step, 1)
                    self.assertEqual(crec.tier, "SSD")
                    self.assertEqual(crec.sub_page, "KEY")
                    self.assertGreater(len(crec.candidate_blocks), 0)
                    self.assertEqual(crec.byte_size, len(crec.candidate_blocks) * 4096)
                    self.assertTrue(crec.is_read)

                elif crec.operation == "TOPK_FETCH":
                    self.assertGreaterEqual(crec.step, 1)
                    self.assertEqual(crec.tier, "SSD")
                    self.assertEqual(crec.sub_page, "VALUE")
                    self.assertEqual(crec.byte_size, 4096)
                    self.assertTrue(crec.is_read)

        self.assertEqual(total_events, 7872)


if __name__ == "__main__":
    unittest.main()
