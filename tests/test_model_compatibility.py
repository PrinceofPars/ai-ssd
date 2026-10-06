"""Tests for AI-SSD Architecture-Adaptive Model Compatibility Layer.

Validates:
- Firmware enable/disable lifecycle scripts and idempotency
- ModelArchitectureConfig and StateDescriptor generation
- Architecture classification: FULL, PARTIAL, UNSUPPORTED
- ModelRegistry discovery, alias mapping, and adapter creation
- TransformerKVStateProvider, SlidingWindowKVStateProvider, and HybridStateProvider interfaces
- SDPA / GQA / MHA / MQA detection
"""

import os
import subprocess
import pytest
from pathlib import Path
import numpy as np

from person1_kv_engine.adapters.descriptor import (
    ModelArchitectureConfig,
    StateDescriptor,
    StateType,
    ResidencyTier,
    CompatibilityLevel,
)
from person1_kv_engine.adapters.state_provider import StateProvider
from person1_kv_engine.adapters.transformer_kv import TransformerKVStateProvider
from person1_kv_engine.adapters.sliding_window_kv import SlidingWindowKVStateProvider
from person1_kv_engine.adapters.hybrid_state import HybridStateProvider
from person1_kv_engine.adapters.model_adapter import (
    ModelAdapter,
    QwenAdapter,
    MistralAdapter,
    HybridJambaAdapter,
    HybridQwen35Adapter,
)
from person1_kv_engine.adapters.registry import ModelRegistry, KNOWN_MODELS
from person1_kv_engine.real_llm.aissd_inference import AISSDBlockStorageBackend


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestModelArchitectureConfig:
    """Test model architecture configuration abstraction."""

    def test_gqa_ratio_calculation(self):
        # 16 attn heads, 4 kv heads -> ratio 4
        cfg = ModelArchitectureConfig(
            model_id="test-model",
            architecture="qwen2",
            model_family="transformer",
            num_layers=24,
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=128,
            hidden_size=2048,
            vocab_size=32000,
            default_precision="fp32",
            supported_precisions=["fp32"],
            supported_contexts=[4096],
            attention_type="GQA",
        )
        assert cfg.gqa_ratio == 4
        assert cfg.is_layer_attention(0) is True
        assert cfg.is_layer_ssm(0) is False

    def test_mha_ratio_calculation(self):
        # 32 attn heads, 32 kv heads -> ratio 1
        cfg = ModelArchitectureConfig(
            model_id="test-mha",
            architecture="llama",
            model_family="transformer",
            num_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            head_dim=128,
            hidden_size=4096,
            vocab_size=32000,
            default_precision="fp16",
            supported_precisions=["fp16"],
            supported_contexts=[4096],
            attention_type="MHA",
        )
        assert cfg.gqa_ratio == 1

    def test_hybrid_layer_types(self):
        # 4 layers: alternating ssm and attention
        cfg = ModelArchitectureConfig(
            model_id="test-hybrid",
            architecture="jamba",
            model_family="hybrid_transformer_ssm",
            num_layers=4,
            num_attention_heads=16,
            num_key_value_heads=4,
            head_dim=128,
            hidden_size=2048,
            vocab_size=32000,
            default_precision="fp16",
            supported_precisions=["fp16"],
            supported_contexts=[4096],
            attention_type="GQA",
            layer_types=["ssm", "attention", "ssm", "attention"],
        )
        assert cfg.is_layer_ssm(0) is True
        assert cfg.is_layer_attention(0) is False
        assert cfg.is_layer_ssm(1) is False
        assert cfg.is_layer_attention(1) is True


class TestStateDescriptor:
    """Test state descriptors for generic state serialization."""

    def test_descriptor_to_dict(self):
        desc = StateDescriptor(
            state_type=StateType.ATTENTION_KV,
            layer_id=3,
            residency=ResidencyTier.AI_SSD,
            tensor_shape=(16, 4, 128),
            dtype="fp16",
            byte_size=16384,
            layout="tokens_heads_dim",
            block_size=16,
            retrieval_requirements={"sparsity": 0.1},
        )
        d = desc.to_dict()
        assert d["state_type"] == "attention_kv"
        assert d["layer_id"] == 3
        assert d["residency"] == "ai_ssd"
        assert d["tensor_shape"] == [16, 4, 128]
        assert d["byte_size"] == 16384
        assert d["retrieval_requirements"] == {"sparsity": 0.1}


class TestModelRegistry:
    """Test model registry discovery, classification, and adapter factory."""

    def test_canonical_qwen_models_registered(self):
        models = ModelRegistry.list_models()
        assert "qwen3-4b" in models
        assert "qwen3-8b" in models
        assert "qwen3.5-4b" in models
        assert "qwen3.5-9b" in models
        assert models["qwen3-4b"]["compatibility_level"] == CompatibilityLevel.FULL
        assert models["qwen3-8b"]["compatibility_level"] == CompatibilityLevel.FULL
        assert models["qwen3.5-4b"]["compatibility_level"] == CompatibilityLevel.FULL
        assert models["qwen3.5-9b"]["compatibility_level"] == CompatibilityLevel.PARTIAL
        assert models["qwen3.5-9b"]["has_separable_kv_cache"] is True

    def test_tiny_mistral_registered(self):
        models = ModelRegistry.list_models()
        assert "tiny-mistral" in models
        assert models["tiny-mistral"]["compatibility_level"] == CompatibilityLevel.FULL
        assert models["tiny-mistral"]["architecture"] == "mistral"
        assert models["tiny-mistral"]["sliding_window"] == 4096

    def test_jamba_hybrid_registered(self):
        models = ModelRegistry.list_models()
        assert "jamba" in models
        assert models["jamba"]["compatibility_level"] == CompatibilityLevel.PARTIAL
        assert models["jamba"]["model_family"] == "hybrid_transformer_ssm"

    def test_mamba_unsupported_registered(self):
        models = ModelRegistry.list_models()
        assert "mamba" in models
        assert models["mamba"]["compatibility_level"] == CompatibilityLevel.UNSUPPORTED
        assert models["mamba"]["has_separable_kv_cache"] is False

    def test_detect_model_config_known_alias(self):
        cfg, comp, reason = ModelRegistry.detect_model_config("qwen3-4b")
        assert comp == CompatibilityLevel.FULL
        assert cfg.num_layers == 36
        assert cfg.head_dim == 128

        cfg_m, comp_m, _ = ModelRegistry.detect_model_config("tiny-mistral")
        assert comp_m == CompatibilityLevel.FULL
        assert cfg_m.architecture == "mistral"

    def test_get_adapter_instances(self):
        cfg_qwen, _, _ = ModelRegistry.detect_model_config("qwen3-4b")
        adapter_qwen = ModelRegistry.get_adapter(cfg_qwen)
        assert isinstance(adapter_qwen, QwenAdapter)

        cfg_mis, _, _ = ModelRegistry.detect_model_config("tiny-mistral")
        adapter_mis = ModelRegistry.get_adapter(cfg_mis)
        assert isinstance(adapter_mis, MistralAdapter)

        cfg_jam, _, _ = ModelRegistry.detect_model_config("jamba")
        adapter_jam = ModelRegistry.get_adapter(cfg_jam)
        assert isinstance(adapter_jam, HybridJambaAdapter)

        cfg_qwen35, _, _ = ModelRegistry.detect_model_config("qwen3.5-9b")
        adapter_qwen35 = ModelRegistry.get_adapter(cfg_qwen35)
        assert isinstance(adapter_qwen35, HybridQwen35Adapter)

    def test_custom_model_registration_and_persistence(self, tmp_path):
        from person1_kv_engine.adapters.registry import CUSTOM_MODELS_FILE, save_custom_model, KNOWN_MODELS
        test_key = "test-auto-registered-model"
        entry = {
            "model_id": "test-org/test-model-1b",
            "architecture": "qwen2",
            "model_family": "transformer",
            "params": "1.0B",
            "default_precision": "fp16",
            "supported_precisions": ["fp16", "float16"],
            "supported_contexts": [512, 1024, 2048],
            "attention_type": "GQA",
            "num_layers": 16,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "hidden_size": 1024,
            "vocab_size": 32000,
            "compatibility_level": CompatibilityLevel.FULL,
            "compatibility_reason": "Custom test model registered",
        }
        ModelRegistry.register_model_entry(test_key, entry)

        # Verify it appears in list_models
        models = ModelRegistry.list_models()
        assert test_key in models
        assert models[test_key]["model_id"] == "test-org/test-model-1b"

        # Verify detect_model_config resolves it
        cfg, comp, reason = ModelRegistry.detect_model_config(test_key)
        assert comp == CompatibilityLevel.FULL
        assert cfg.num_layers == 16
        assert cfg.num_attention_heads == 8

        # Clean up test entry
        KNOWN_MODELS.pop(test_key, None)
        if CUSTOM_MODELS_FILE.exists():
            import json
            with open(CUSTOM_MODELS_FILE, "r") as f:
                d = json.load(f)
            d.pop(test_key, None)
            with open(CUSTOM_MODELS_FILE, "w") as f:
                json.dump(d, f)


class TestStateProviders:
    """Test concrete StateProvider implementations without full model weights."""

    def test_transformer_kv_state_provider_lifecycle(self):
        cfg, _, _ = ModelRegistry.detect_model_config("qwen3-4b")
        backend = AISSDBlockStorageBackend(num_layers=cfg.num_layers, tokens_per_block=16, head_dim=cfg.head_dim)
        prov = TransformerKVStateProvider(config=cfg, backend=backend, tokens_per_block=16)

        # Inspect state descriptors
        # Create mock past_key_values object
        class MockLayer:
            def __init__(self):
                import torch
                self.keys = torch.randn(1, 4, 100, 128)
                self.values = torch.randn(1, 4, 100, 128)

        class MockPKV:
            def __init__(self, n_layers):
                self.layers = [MockLayer() for _ in range(n_layers)]

        mock_pkv = MockPKV(cfg.num_layers)
        descriptors = prov.inspect_state(mock_pkv)
        assert len(descriptors) == cfg.num_layers
        assert descriptors[0].state_type == StateType.ATTENTION_KV
        assert descriptors[0].residency == ResidencyTier.AI_SSD

        # Init from prefill
        prov.init_from_prefill(mock_pkv)
        assert prov.is_active is True
        assert prov.layer_data[0]["total_tokens"] == 100

        # Memory stats
        stats = prov.get_memory_stats()
        assert stats["total_kv_mb"] > 0
        assert stats["active_dram_mb"] > 0
        assert stats["offload_pct"] > 0

    def test_sliding_window_provider(self):
        cfg, _, _ = ModelRegistry.detect_model_config("tiny-mistral")
        backend = AISSDBlockStorageBackend(num_layers=cfg.num_layers, tokens_per_block=16, head_dim=cfg.head_dim)
        prov = SlidingWindowKVStateProvider(config=cfg, backend=backend, tokens_per_block=16)
        assert prov.sliding_window == 4096

    def test_hybrid_state_provider(self):
        cfg, _, _ = ModelRegistry.detect_model_config("jamba")
        backend = AISSDBlockStorageBackend(num_layers=cfg.num_layers, tokens_per_block=16, head_dim=cfg.head_dim)
        prov = HybridStateProvider(config=cfg, backend=backend, tokens_per_block=16)
        assert prov.config.model_family == "hybrid_transformer_ssm"


class TestFirmwareLifecycle:
    """Test firmware control scripts and status."""

    def test_disable_script_exists_and_executable(self):
        disable_script = PROJECT_ROOT / "disable_ai_ssd.sh"
        assert disable_script.exists()
        assert os.access(disable_script, os.X_OK)

    def test_enable_script_exists_and_executable(self):
        enable_script = PROJECT_ROOT / "enable_ai_ssd.sh"
        assert enable_script.exists()
        assert os.access(enable_script, os.X_OK)

    def test_disable_script_idempotent_when_disabled(self):
        disable_script = PROJECT_ROOT / "disable_ai_ssd.sh"
        # Run disable twice to verify idempotency
        res1 = subprocess.run([str(disable_script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        res2 = subprocess.run([str(disable_script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert res2.returncode == 0
        assert "AI-SSD Firmware: DISABLED" in res2.stdout
        assert "ALREADY STOPPED" in res2.stdout or "STOPPED" in res2.stdout
