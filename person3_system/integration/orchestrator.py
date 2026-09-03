"""
System Orchestrator: Binds KV Engine, SSD backend (or mocks), and Prefetcher into a unified pipeline.
"""

from typing import Any
from person3_system.api.ai_ssd import AISSDSystem
from person3_system.prefetch.prefetcher import SpeculativePrefetcher


class SystemOrchestrator:
    def __init__(self, kv_engine: Any, ssd_engine: Any, enable_prefetch: bool = True):
        self.kv_engine = kv_engine
        self.ssd_engine = ssd_engine
        self.prefetcher = SpeculativePrefetcher() if enable_prefetch else None
        self.api = AISSDSystem(
            kv_engine=self.kv_engine,
            ssd_engine=self.ssd_engine,
            prefetcher=self.prefetcher,
        )
