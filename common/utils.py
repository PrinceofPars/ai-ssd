"""
Common utilities for configuration loading, I/O, and timing.
"""

from pathlib import Path
from typing import Any, Dict
import json
import yaml


def load_yaml(file_path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_json(data: Dict[str, Any], file_path: str | Path) -> None:
    """Save dictionary to a formatted JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(file_path: str | Path) -> Dict[str, Any]:
    """Load a JSON file."""
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_kv_cache_size_bytes(
    context_length: int,
    layers: int,
    heads: int,
    head_dim: int,
    dtype: str = "FP16",
) -> int:
    """
    Calculate the total uncompressed KV cache size for an LLM sequence.
    Size = 2 (K & V) * layers * heads * head_dim * context_length * bytes_per_elem
    """
    bytes_per_elem = 2 if dtype.upper() == "FP16" else 1
    return 2 * layers * heads * head_dim * context_length * bytes_per_elem
