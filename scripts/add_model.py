#!/usr/bin/env python3
"""
AI-SSD V2 — Automated Model Onboarding & Registration CLI
Validates, downloads (if not already cached), inspects architecture, and registers new models into AI-SSD.

Usage:
    python scripts/add_model.py <model_name_or_hf_id> [--model-key <key>] [--precision fp16|fp32]
"""

import sys
import os
import glob
import re
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from person1_kv_engine.adapters.registry import ModelRegistry
from person1_kv_engine.adapters.descriptor import CompatibilityLevel


def sanitize_model_key(raw_name: str) -> str:
    """Generates a clean slugified model-key from a model name or HF ID."""
    name = raw_name.strip()
    if "/" in name:
        name = name.split("/")[-1]
    name = re.sub(r"[^a-zA-Z0-9._-]", "-", name)
    name = re.sub(r"-+", "-", name).strip("-").lower()
    return name


def resolve_model_identifier(raw_name: str) -> str:
    """Resolves short names or validates Hugging Face repository IDs."""
    name = raw_name.strip()

    # If it's a local directory or already has org prefix
    if os.path.isdir(name) or "/" in name:
        return name

    # Check if already registered
    all_models = ModelRegistry.list_models()
    if name.lower() in all_models:
        return all_models[name.lower()]["model_id"]

    # Try common vendor prefixes
    vendor_candidates = [
        f"Qwen/{name}",
        f"mistralai/{name}",
        f"meta-llama/{name}",
        f"google/{name}",
        f"openaccess-ai-collective/{name}",
    ]

    try:
        from transformers import AutoConfig
        for cand in vendor_candidates:
            try:
                AutoConfig.from_pretrained(cand)
                return cand
            except Exception:
                pass
    except ImportError:
        pass

    return name


def check_model_cache(model_id: str) -> Tuple[bool, Optional[str], List[str]]:
    """Checks whether the model config and weights are already present locally."""
    # 1. Local directory check
    if os.path.isdir(model_id):
        weights = (
            glob.glob(os.path.join(model_id, "*.safetensors")) +
            glob.glob(os.path.join(model_id, "*.bin"))
        )
        return True, model_id, weights

    # 2. Hugging Face Hub cache check
    try:
        from huggingface_hub import try_to_load_from_cache
        cfg_path = try_to_load_from_cache(model_id, "config.json")
        if cfg_path and isinstance(cfg_path, str) and os.path.exists(cfg_path):
            snapshot_dir = os.path.dirname(cfg_path)
            weights = (
                glob.glob(os.path.join(snapshot_dir, "*.safetensors")) +
                glob.glob(os.path.join(snapshot_dir, "*.bin"))
            )
            if weights:
                return True, snapshot_dir, weights
            return False, snapshot_dir, []
    except Exception:
        pass

    return False, None, []


def download_model_if_needed(model_id: str, force: bool = False) -> Tuple[bool, Optional[str]]:
    """Downloads model weights and tokenizer from Hugging Face if not already present."""
    if os.path.isdir(model_id):
        print(f"[CACHE HIT] Using local directory: {model_id}")
        return True, model_id

    is_cached, snap_dir, weights = check_model_cache(model_id)

    if is_cached and not force:
        print(f"[CACHE HIT] Model '{model_id}' is already downloaded in local cache ({len(weights)} weight file(s)).")
        print("Skipping download.")
        return True, snap_dir

    print(f"\n[DOWNLOADING] Model '{model_id}' is not in local cache.")
    print("Downloading weights and tokenizer from Hugging Face Hub...")

    try:
        from huggingface_hub import snapshot_download
        download_dir = snapshot_download(
            repo_id=model_id,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx", "*.pt"]
        )
        # Also ensure tokenizer is cached
        try:
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(model_id)
        except Exception:
            pass
        print(f"[DOWNLOAD COMPLETE] Model weights successfully cached at: {download_dir}\n")
        return True, download_dir
    except Exception as e:
        print(f"[ERROR] Failed to download model '{model_id}': {e}")
        return False, None


def estimate_params(cfg: Any, weight_files: List[str]) -> str:
    """Estimates parameter count from config or weight file sizes."""
    # 1. Check weight files total size
    total_bytes = 0
    for wf in weight_files:
        try:
            total_bytes += os.path.getsize(wf)
        except Exception:
            pass

    if total_bytes > 0:
        # Assuming FP16 / BF16 (2 bytes/param)
        params_fp16 = total_bytes / 2.0
        if params_fp16 >= 1e9:
            return f"{params_fp16 / 1e9:.1f}B"
        elif params_fp16 >= 1e6:
            return f"{params_fp16 / 1e6:.1f}M"

    # 2. Estimate from Transformer config
    layers = getattr(cfg, "num_hidden_layers", 0)
    hidden = getattr(cfg, "hidden_size", 0)
    vocab = getattr(cfg, "vocab_size", 0)
    if layers > 0 and hidden > 0:
        # Standard transformer approx: 12 * L * H^2 + V * H
        approx = (12 * layers * (hidden ** 2)) + (vocab * hidden)
        if approx >= 1e9:
            return f"{approx / 1e9:.1f}B"
        elif approx >= 1e6:
            return f"{approx / 1e6:.1f}M"

    return "N/A"


def inspect_and_register_model(
    model_identifier: str,
    custom_model_key: Optional[str] = None,
    precision: Optional[str] = None,
    skip_download: bool = False,
    force_download: bool = False,
) -> Optional[str]:
    """Inspects architecture, downloads weights (if needed), and registers into ModelRegistry."""
    resolved_id = resolve_model_identifier(model_identifier)
    model_key = (custom_model_key.strip().lower() if custom_model_key else sanitize_model_key(resolved_id))

    print("==========================================================================================")
    print("                       AI-SSD MODEL ONBOARDING & REGISTRATION                             ")
    print("==========================================================================================")
    print(f"Target Identifier:     {model_identifier}")
    print(f"Resolved Model ID:     {resolved_id}")
    print(f"Assigned Model Key:    {model_key}")
    print("------------------------------------------------------------------------------------------")

    # Step 1: Validate configuration
    print("[1/3] Validating configuration on Hugging Face Hub / local path...")
    try:
        from transformers import AutoConfig
        hf_cfg = AutoConfig.from_pretrained(resolved_id)
    except Exception as e:
        print(f"\n[ERROR] Could not load configuration for '{resolved_id}': {e}")
        print("Please verify that the repository exists on Hugging Face (e.g., 'Qwen/Qwen2.5-0.5B').")
        print("For private/gated repositories, log in with: huggingface-cli login\n")
        return None

    # Step 2: Download weights if needed
    print("\n[2/3] Checking local cache & model weights...")
    weight_files = []
    if not skip_download:
        ok, snap_path = download_model_if_needed(resolved_id, force=force_download)
        if not ok:
            print("[WARNING] Could not download full weights. Model will be registered for architecture inspection only.")
        elif snap_path and os.path.exists(snap_path):
            weight_files = glob.glob(os.path.join(snap_path, "*.safetensors")) + glob.glob(os.path.join(snap_path, "*.bin"))
    else:
        print("[SKIP] Skipping weight download (--skip-download requested).")

    # Step 3: Inspect architecture
    print("\n[3/3] Inspecting architecture, tensors, and compatibility...")
    text_cfg = getattr(hf_cfg, "text_config", None) or hf_cfg
    model_type = getattr(hf_cfg, "model_type", getattr(text_cfg, "model_type", "")).lower()
    num_layers = getattr(text_cfg, "num_hidden_layers", 0)
    num_attn_heads = getattr(text_cfg, "num_attention_heads", 0)
    num_kv_heads = getattr(text_cfg, "num_key_value_heads", num_attn_heads)
    hidden_size = getattr(text_cfg, "hidden_size", 0)
    head_dim = getattr(text_cfg, "head_dim", (hidden_size // num_attn_heads) if num_attn_heads > 0 else 64)
    vocab_size = getattr(text_cfg, "vocab_size", 0)
    sliding_window = getattr(text_cfg, "sliding_window", None)

    # Determine attention geometry
    if num_kv_heads == 1 and num_attn_heads > 1:
        attn_type = "MQA"
    elif num_kv_heads < num_attn_heads:
        attn_type = f"GQA ({num_attn_heads}:{num_kv_heads})"
    else:
        attn_type = "MHA"

    param_str = estimate_params(hf_cfg, weight_files)
    chosen_precision = (precision or "fp16").lower()

    # Determine compatibility
    if model_type in ("qwen", "qwen2", "qwen3", "qwen3_5"):
        comp_level = CompatibilityLevel.FULL
        comp_reason = f"Supported QWEN Transformer architecture ({attn_type}) via QwenAdapter"
        family = "transformer"
        arch_type = "qwen2"
    elif model_type in ("mistral", "llama", "llama2", "llama3"):
        comp_level = CompatibilityLevel.FULL
        comp_reason = f"Supported {model_type.upper()} Transformer architecture ({attn_type}) via MistralAdapter"
        family = "transformer"
        arch_type = "mistral"
    elif model_type in ("jamba",):
        comp_level = CompatibilityLevel.PARTIAL
        comp_reason = "Hybrid Attention + Mamba: Attention KV offloaded to AI-SSD, SSM resident in DRAM"
        family = "hybrid_transformer_ssm"
        arch_type = "jamba"
    elif model_type in ("mamba", "mamba2", "rwkv"):
        comp_level = CompatibilityLevel.UNSUPPORTED
        comp_reason = f"Pure state-space recurrent model ({model_type}) lacks separable KV cache for attention Top-K offload"
        family = "ssm"
        arch_type = model_type
    elif num_layers > 0 and num_attn_heads > 0:
        comp_level = CompatibilityLevel.FULL
        comp_reason = f"Transformer causal decoder ({model_type}) supported with separable KV cache"
        family = "transformer"
        arch_type = "mistral"
    else:
        comp_level = CompatibilityLevel.UNSUPPORTED
        comp_reason = f"Unsupported architecture ({model_type}). Requires a custom ModelAdapter and StateProvider."
        family = "unknown"
        arch_type = model_type

    # Register into ModelRegistry
    entry = {
        "model_id": resolved_id,
        "architecture": arch_type,
        "model_family": family,
        "params": param_str,
        "default_precision": chosen_precision,
        "supported_precisions": ["fp16", "float16", "fp32", "float32"],
        "supported_contexts": [512, 1024, 2048, 4096, 8192],
        "attention_type": attn_type,
        "sliding_window": sliding_window,
        "num_layers": num_layers,
        "num_attention_heads": num_attn_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "compatibility_level": comp_level,
        "compatibility_reason": comp_reason,
    }

    ModelRegistry.register_model_entry(model_key, entry)

    # Print success summary card
    print("\n==========================================================================================")
    print("                              MODEL REGISTRATION SUMMARY                                  ")
    print("==========================================================================================")
    print(f"Model Key:             {model_key}  <-- USE THIS EXACT KEY FOR DEMO-INFERENCE")
    print(f"Hugging Face ID:       {resolved_id}")
    print(f"Architecture Family:   {family.upper()}")
    print(f"Architecture Type:     {arch_type}")
    print(f"Estimated Parameters:  {param_str}")
    print(f"Attention Type:        {attn_type}")
    print(f"Layers:                {num_layers}")
    print(f"Attention Heads:       {num_attn_heads}")
    print(f"KV Heads:              {num_kv_heads}")
    print(f"Head Dimension:        {head_dim}")
    print(f"Hidden Size:           {hidden_size}")
    print(f"Sliding Window:        {sliding_window or 'None'}")
    print(f"Default Precision:     {chosen_precision.upper()}")
    print(f"Compatibility Level:   {comp_level.value}")
    print(f"Compatibility Reason:  {comp_reason}")
    print("==========================================================================================")
    print("\nNext steps:")
    print(f"  1. View in registry:       python scripts/demo_inference.py --list-models")
    print(f"  2. Inspect architecture:   python scripts/demo_inference.py --inspect-model {model_key}")
    print(f"  3. Run live inference:     python scripts/demo_inference.py --model {model_key} --context 4096\n")

    return model_key


def main():
    parser = argparse.ArgumentParser(
        description="AI-SSD V2 — Automated Model Onboarding & Registration CLI",
        add_help=True
    )
    parser.add_argument("model", type=str, help="Hugging Face repository ID (e.g., 'Qwen/Qwen2.5-0.5B') or local path")
    parser.add_argument("--model-key", type=str, default=None, help="Custom short key to register under (e.g., 'my-model')")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "fp32"], help="Default precision (fp16 or fp32)")
    parser.add_argument("--skip-download", action="store_true", help="Inspect and register without downloading full weights")
    parser.add_argument("--force-download", action="store_true", help="Force re-download even if already present in cache")

    args = parser.parse_args()
    res = inspect_and_register_model(
        model_identifier=args.model,
        custom_model_key=args.model_key,
        precision=args.precision,
        skip_download=args.skip_download,
        force_download=args.force_download,
    )
    if res is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
