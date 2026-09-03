"""
System Orchestrator: Binds Person 1 KV Engine, Person 2 Physical Flash SSD,
and Person 3 Speculative Prefetcher into a cohesive production pipeline.
"""

from typing import Any, Optional, Dict, List
import numpy as np

from person3_system.api.ai_ssd import AISSDSystem
from person3_system.prefetch.prefetcher import SpeculativePrefetcher
from person3_system.adapters.physical_adapter import PhysicalTensorAwareStorageAdapter
from person1_kv_engine.baseline.transformer_attention import AttentionConfig
from person1_kv_engine.tiering.hot_cold_classifier import TieringPolicy
from person1_kv_engine.tiering.tiered_kv_manager import TieredKVManager
from person1_kv_engine.computational_storage.instorage_pruner import InStorageAttentionPruner


class SystemOrchestrator:
    """
    Central coordinator orchestrating real Person 1, Person 2, and Person 3 subsystems.
    Can be initialized with custom instances or defaults to fully wired physical subsystems.
    """

    def __init__(
        self,
        kv_engine: Optional[Any] = None,
        ssd_engine: Optional[Any] = None,
        enable_prefetch: bool = True,
        mode: str = "tensor_aware",
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        dtype: str = "FP16",
    ):
        self.mode = mode
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # 1. Wire Storage Engine: PhysicalTensorAwareStorageAdapter by default
        if ssd_engine is not None:
            self.ssd_engine = ssd_engine
        else:
            self.ssd_engine = PhysicalTensorAwareStorageAdapter(
                mode=mode,
                channels=8,
                controller_dram_size_mb=1024,
            )

        # 2. Wire Prefetcher
        self.prefetcher = SpeculativePrefetcher(
            buffer_capacity_blocks=512,
            bytes_per_block=4096,
            gpu_compute_time_per_layer_us=65.0,
        ) if enable_prefetch else None

        # 3. Wire KV Engine: TieredKVManager by default
        if kv_engine is not None:
            self.kv_engine = kv_engine
            self.pruner = None
        else:
            self.config = AttentionConfig(
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                dtype=np.float16 if dtype == "FP16" else np.float32,
            )
            self.policy = TieringPolicy(
                sink_tokens=64,
                sliding_window_tokens=512,
                tokens_per_block=16,
            )
            self.kv_engine = TieredKVManager(
                config=self.config,
                storage=self.ssd_engine,
                policy=self.policy,
            )
            self.pruner = InStorageAttentionPruner(
                config=self.config,
                storage=self.ssd_engine,
                manager=self.kv_engine,
                default_top_k=16,
            )

        # 4. Wire Unified API Gateway
        self.api = AISSDSystem(
            kv_engine=self.kv_engine,
            ssd_engine=self.ssd_engine,
            prefetcher=self.prefetcher,
        )

        # Register access listener so prefetcher can observe storage events
        if self.prefetcher and hasattr(self.ssd_engine, "register_access_listener"):
            def on_access(event: Dict[str, Any]):
                op = event.get("op")
                if op in ("PREFETCH_KV", "PREFETCH"):
                    return  # Prevent recursive prefetch cascade
                bids = event.get("block_ids", [])
                lid = event.get("layer_id", 0)
                if bids:
                    self.prefetcher.prefetch_next_layer(
                        current_layer_id=lid,
                        active_block_ids=bids,
                        storage_backend=None,  # Avoid re-triggering storage listener
                    )
            self.ssd_engine.register_access_listener(on_access)

    def get_telemetry(self) -> Dict[str, Any]:
        """Collects combined telemetry across all subsystems."""
        telemetry = {
            "mode": getattr(self.ssd_engine, "mode", self.mode),
        }
        if hasattr(self.ssd_engine, "get_telemetry"):
            telemetry["storage"] = self.ssd_engine.get_telemetry()
        if self.prefetcher:
            telemetry["prefetch"] = self.prefetcher.get_telemetry()
        if hasattr(self.kv_engine, "get_host_ram_usage_mb"):
            telemetry["host_ram_mb"] = self.kv_engine.get_host_ram_usage_mb()
            telemetry["ssd_storage_mb"] = self.kv_engine.get_offloaded_storage_mb()
        return telemetry
