"""Model Adapter Registry and Architecture Detection for AI-SSD.

Discovers, classifies, and instantiates ModelAdapters for Qwen, Mistral, LLaMA,
hybrid (Jamba), and unsupported architectures.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Type, Tuple
import logging
import json
from pathlib import Path

from person1_kv_engine.adapters.descriptor import (
    ModelArchitectureConfig,
    CompatibilityLevel,
)
from person1_kv_engine.adapters.model_adapter import (
    ModelAdapter,
    QwenAdapter,
    MistralAdapter,
    HybridJambaAdapter,
    HybridQwen35Adapter,
)

logger = logging.getLogger(__name__)

CUSTOM_MODELS_FILE = Path(__file__).resolve().parent / "custom_models.json"


def load_custom_models() -> Dict[str, Dict[str, Any]]:
    """Loads user-registered models from custom_models.json."""
    if not CUSTOM_MODELS_FILE.exists():
        return {}
    try:
        with open(CUSTOM_MODELS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if "compatibility_level" in v and isinstance(v["compatibility_level"], str):
                try:
                    v["compatibility_level"] = CompatibilityLevel(v["compatibility_level"])
                except Exception:
                    pass
        return data
    except Exception as e:
        logger.warning(f"Failed to load custom models: {e}")
        return {}


def save_custom_model(model_key: str, entry: Dict[str, Any]) -> None:
    """Saves a user-registered model to custom_models.json and updates in-memory registry."""
    current = {}
    if CUSTOM_MODELS_FILE.exists():
        try:
            with open(CUSTOM_MODELS_FILE, "r", encoding="utf-8") as f:
                current = json.load(f)
        except Exception:
            current = {}
    entry_copy = dict(entry)
    if "compatibility_level" in entry_copy and hasattr(entry_copy["compatibility_level"], "value"):
        entry_copy["compatibility_level"] = entry_copy["compatibility_level"].value
    current[model_key] = entry_copy
    with open(CUSTOM_MODELS_FILE, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)
    KNOWN_MODELS[model_key] = entry


# Built-in canonical validated models registry
KNOWN_MODELS: Dict[str, Dict[str, Any]] = {
    "qwen3-4b": {
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "architecture": "qwen2",
        "model_family": "transformer",
        "params": "4.02B",
        "default_precision": "fp32",
        "supported_precisions": ["fp32", "float32"],
        "supported_contexts": [512, 1024, 2048, 4096, 8192, 16384, 32768],
        "attention_type": "GQA",
        "num_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "hidden_size": 2048,
        "vocab_size": 151936,
        "compatibility_level": CompatibilityLevel.FULL,
        "compatibility_reason": "Validated canonical V2 release",
        "dense_peak_rss_mb": 17028.4,
        "expected_tokens_seed42": [
            11773, 48758, 6529, 19826, 4712, 57203, 12756, 3871,
            1948, 279, 3239, 4621, 323, 9144, 6894, 13
        ],
    },
    "qwen3-8b": {
        "model_id": "Qwen/Qwen3-8B",
        "architecture": "qwen2",
        "model_family": "transformer",
        "params": "8.19B",
        "default_precision": "fp16",
        "supported_precisions": ["fp16", "float16"],
        "supported_contexts": [512, 1024, 2048, 4096, 8192, 16384],
        "attention_type": "GQA",
        "num_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 4096,
        "vocab_size": 152064,
        "compatibility_level": CompatibilityLevel.FULL,
        "compatibility_reason": "Validated canonical V2 release",
        "dense_peak_rss_mb": 18022.5,
        "expected_tokens_seed42": [
            11773, 48758, 6529, 19826, 4712, 57203, 12756, 3871,
            1948, 279, 3239, 4621, 323, 9144, 6894, 13
        ],
    },
    "qwen2.5-0.5b": {
        "model_id": "Qwen/Qwen2.5-0.5B",
        "architecture": "qwen2",
        "model_family": "transformer",
        "params": "0.49B",
        "default_precision": "fp32",
        "supported_precisions": ["fp32", "float32"],
        "supported_contexts": [512, 1024, 2048, 4096],
        "attention_type": "GQA",
        "num_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "hidden_size": 896,
        "vocab_size": 151936,
        "compatibility_level": CompatibilityLevel.FULL,
        "compatibility_reason": "Validated development model",
    },
    "qwen3.5-4b": {
        "model_id": "Qwen/Qwen3.5-4B-Instruct",
        "architecture": "qwen2",
        "model_family": "transformer",
        "params": "4.0B",
        "default_precision": "fp32",
        "supported_precisions": ["fp32", "float32", "fp16", "float16"],
        "supported_contexts": [512, 1024, 2048, 4096, 8192],
        "attention_type": "GQA",
        "num_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "hidden_size": 2048,
        "vocab_size": 151936,
        "compatibility_level": CompatibilityLevel.FULL,
        "compatibility_reason": "Qwen 3.5 GQA Transformer architecture supported via QwenAdapter",
    },
    "qwen3.5-9b": {
        "model_id": "Qwen/Qwen3.5-9B",
        "architecture": "qwen3_5",
        "model_family": "hybrid_recurrent",
        "params": "9.7B",
        "default_precision": "fp16",
        "supported_precisions": ["fp16", "float16"],
        "supported_contexts": [512, 1024, 2048, 4096, 8192],
        "attention_type": "Hybrid (24 Linear SSM + 8 Full GQA)",
        "num_layers": 32,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "hidden_size": 4096,
        "vocab_size": 248320,
        "has_separable_kv_cache": True,
        "compatibility_level": CompatibilityLevel.PARTIAL,
        "compatibility_reason": "Hybrid Attention + Linear SSM: 8 Full GQA Attention layers offloaded to AI-SSD, 24 Linear Attention SSM layers resident in DRAM",
    },
    "tiny-mistral": {
        "model_id": "openaccess-ai-collective/tiny-mistral",
        "architecture": "mistral",
        "model_family": "transformer",
        "params": "0.21B",
        "default_precision": "fp32",
        "supported_precisions": ["fp32", "float32", "fp16", "float16"],
        "supported_contexts": [512, 1024, 2048, 4096],
        "attention_type": "GQA",
        "sliding_window": 4096,
        "num_layers": 8,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 32,
        "hidden_size": 512,
        "vocab_size": 32000,
        "compatibility_level": CompatibilityLevel.FULL,
        "compatibility_reason": "Validated Mistral architecture with GQA and sliding-window support",
    },
    "mistral-7b": {
        "model_id": "mistralai/Mistral-7B-v0.1",
        "architecture": "mistral",
        "model_family": "transformer",
        "params": "7.24B",
        "default_precision": "fp16",
        "supported_precisions": ["fp16", "float16"],
        "supported_contexts": [512, 1024, 2048, 4096, 8192],
        "attention_type": "GQA",
        "sliding_window": 4096,
        "num_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 4096,
        "vocab_size": 32000,
        "compatibility_level": CompatibilityLevel.FULL,
        "compatibility_reason": "Mistral 7B architecture supported via MistralAdapter",
    },
    "jamba": {
        "model_id": "ai21labs/AI21-Jamba-1.5-Mini",
        "architecture": "jamba",
        "model_family": "hybrid_transformer_ssm",
        "params": "12B (52B total MoE)",
        "default_precision": "fp16",
        "supported_precisions": ["fp16", "float16", "bf16"],
        "supported_contexts": [512, 1024, 2048, 4096],
        "attention_type": "GQA",
        "num_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 4096,
        "vocab_size": 65536,
        "compatibility_level": CompatibilityLevel.PARTIAL,
        "compatibility_reason": "Hybrid Attention + Mamba: Attention KV offloaded to AI-SSD, SSM states resident in DRAM",
    },
    "mamba": {
        "model_id": "state-spaces/mamba-130m-hf",
        "architecture": "mamba",
        "model_family": "ssm",
        "params": "0.13B",
        "default_precision": "fp32",
        "supported_precisions": ["fp32", "float32"],
        "supported_contexts": [512, 1024, 2048],
        "attention_type": "None (Pure State-Space)",
        "num_layers": 24,
        "num_attention_heads": 0,
        "num_key_value_heads": 0,
        "head_dim": 0,
        "hidden_size": 768,
        "vocab_size": 50280,
        "has_separable_kv_cache": False,
        "compatibility_level": CompatibilityLevel.UNSUPPORTED,
        "compatibility_reason": "Pure state-space recurrent model lacks separable KV cache for attention Top-K offload",
    },
}


class ModelRegistry:
    """Registry managing model adapters and architecture detection."""

    _adapter_classes: Dict[str, Type[ModelAdapter]] = {
        "qwen2": QwenAdapter,
        "qwen": QwenAdapter,
        "mistral": MistralAdapter,
        "llama": MistralAdapter,  # LLaMA shares rotary and SDPA attention interface with Mistral
        "jamba": HybridJambaAdapter,
        "qwen3_5": HybridQwen35Adapter,
    }

    @classmethod
    def register_adapter(cls, arch_name: str, adapter_cls: Type[ModelAdapter]) -> None:
        cls._adapter_classes[arch_name.lower()] = adapter_cls

    @classmethod
    def detect_model_config(cls, model_identifier: str) -> Tuple[ModelArchitectureConfig, CompatibilityLevel, str]:
        """Inspects a model identifier or Hugging Face config and creates a ModelArchitectureConfig."""
        norm_key = model_identifier.strip().lower()

        # 1. Check known aliases
        if norm_key in ("qwen3.5", "qwen-3.5"):
            norm_key = "qwen3.5-4b"
        elif norm_key in ("qwen3.5-9b", "qwen-3.5-9b", "qwen3.5_9b", "qwen/qwen3.5-9b"):
            norm_key = "qwen3.5-9b"

        all_models = cls.list_models()
        if norm_key in all_models:
            entry = all_models[norm_key]
            cfg = ModelArchitectureConfig(
                model_id=entry["model_id"],
                architecture=entry["architecture"],
                model_family=entry["model_family"],
                num_layers=entry["num_layers"],
                num_attention_heads=entry["num_attention_heads"],
                num_key_value_heads=entry["num_key_value_heads"],
                head_dim=entry["head_dim"],
                hidden_size=entry["hidden_size"],
                vocab_size=entry["vocab_size"],
                default_precision=entry["default_precision"],
                supported_precisions=entry["supported_precisions"],
                supported_contexts=entry["supported_contexts"],
                attention_type=entry["attention_type"],
                sliding_window=entry.get("sliding_window"),
                compatibility_level=entry["compatibility_level"],
                compatibility_reason=entry["compatibility_reason"],
                param_count=entry.get("params"),
                expected_tokens_seed42=entry.get("expected_tokens_seed42"),
                dense_peak_rss_mb=entry.get("dense_peak_rss_mb"),
            )
            return cfg, cfg.compatibility_level, cfg.compatibility_reason

        # 2. Try loading AutoConfig from Hugging Face
        try:
            from transformers import AutoConfig
            try:
                hf_cfg = AutoConfig.from_pretrained(model_identifier)
            except Exception:
                # If identifier lacks vendor namespace (e.g. 'qwen3.5-9b'), try standard org prefixes
                if "/" not in model_identifier:
                    alt_id = None
                    if norm_key.startswith("qwen"):
                        alt_id = f"Qwen/{model_identifier}"
                    elif norm_key.startswith("mistral"):
                        alt_id = f"mistralai/{model_identifier}"
                    if alt_id:
                        hf_cfg = AutoConfig.from_pretrained(alt_id)
                        model_identifier = alt_id
                    else:
                        raise
                else:
                    raise
        except Exception as e:
            # Cannot inspect
            cfg = ModelArchitectureConfig(
                model_id=model_identifier,
                architecture="unknown",
                model_family="unknown",
                num_layers=0,
                num_attention_heads=0,
                num_key_value_heads=0,
                head_dim=0,
                hidden_size=0,
                vocab_size=0,
                default_precision="fp16",
                supported_precisions=[],
                supported_contexts=[],
                attention_type="unknown",
                has_separable_kv_cache=False,
                compatibility_level=CompatibilityLevel.UNSUPPORTED,
                compatibility_reason=f"Failed to inspect configuration: {e}",
            )
            return cfg, CompatibilityLevel.UNSUPPORTED, cfg.compatibility_reason

        text_cfg = getattr(hf_cfg, "text_config", None) or hf_cfg
        model_type = getattr(hf_cfg, "model_type", getattr(text_cfg, "model_type", "")).lower()
        num_layers = getattr(text_cfg, "num_hidden_layers", 0)
        num_attn_heads = getattr(text_cfg, "num_attention_heads", 0)
        num_kv_heads = getattr(text_cfg, "num_key_value_heads", num_attn_heads)
        hidden_size = getattr(text_cfg, "hidden_size", 0)
        head_dim = getattr(text_cfg, "head_dim", (hidden_size // num_attn_heads) if num_attn_heads > 0 else 64)
        vocab_size = getattr(text_cfg, "vocab_size", 0)
        sliding_window = getattr(text_cfg, "sliding_window", None)

        if num_kv_heads == 1 and num_attn_heads > 1:
            attn_type = "MQA"
        elif num_kv_heads < num_attn_heads:
            attn_type = "GQA"
        else:
            attn_type = "MHA"

        # Determine compatibility
        if model_type in ("qwen2", "qwen", "qwen3", "mistral", "llama"):
            comp_level = CompatibilityLevel.FULL
            reason = f"Supported {model_type.upper()} Transformer architecture ({attn_type})"
            family = "transformer"
        elif model_type in ("qwen3_5",):
            comp_level = CompatibilityLevel.PARTIAL
            reason = f"Hybrid architecture ({model_type}): Attention KV layers offloaded to AI-SSD, Linear Attention (DeltaNet/SSM) layers resident in DRAM."
            family = "hybrid_recurrent"
        elif model_type in ("jamba",):
            comp_level = CompatibilityLevel.PARTIAL
            reason = "Hybrid Attention + Mamba architecture: Attention KV offloaded to AI-SSD, SSM resident in DRAM"
            family = "hybrid_transformer_ssm"
        elif model_type in ("mamba", "mamba2", "rwkv"):
            comp_level = CompatibilityLevel.UNSUPPORTED
            reason = f"Pure state-space/recurrent model ({model_type}) lacks separable KV cache for attention Top-K offload"
            family = "ssm"
        else:
            comp_level = CompatibilityLevel.UNSUPPORTED
            reason = f"Unsupported architecture family: {model_type}. Requires a custom ModelAdapter and StateProvider."
            family = "unknown"

        cfg = ModelArchitectureConfig(
            model_id=model_identifier,
            architecture=model_type,
            model_family=family,
            num_layers=num_layers,
            num_attention_heads=num_attn_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            default_precision="fp16",
            supported_precisions=["fp16", "float16", "fp32", "float32"],
            supported_contexts=[512, 1024, 2048, 4096],
            attention_type=attn_type,
            sliding_window=sliding_window,
            compatibility_level=comp_level,
            compatibility_reason=reason,
        )
        return cfg, comp_level, reason

    @classmethod
    def get_adapter(cls, config: ModelArchitectureConfig) -> Optional[ModelAdapter]:
        """Returns the appropriate ModelAdapter instance for the given config."""
        arch = config.architecture.lower()
        adapter_cls = cls._adapter_classes.get(arch)
        if adapter_cls is None:
            # Fallback check on model family
            if config.model_family == "transformer":
                adapter_cls = MistralAdapter
            elif config.model_family == "hybrid_transformer_ssm":
                adapter_cls = HybridJambaAdapter
            else:
                return None
        return adapter_cls(config)

    @classmethod
    def register_model_entry(cls, model_key: str, entry: Dict[str, Any]) -> None:
        """Persists and registers a new model entry into the registry."""
        save_custom_model(model_key, entry)

    @classmethod
    def list_models(cls) -> Dict[str, Dict[str, Any]]:
        """Returns all registered models with their metadata and compatibility status."""
        models = dict(load_custom_models())
        models.update(KNOWN_MODELS)
        return models
