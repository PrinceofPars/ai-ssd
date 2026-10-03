"""Speculative prefetcher package."""
from person3_system.prefetch.prefetcher import SpeculativePrefetcher
from person3_system.prefetch.v2_prefetcher import V2Prefetcher
from person3_system.prefetch.inference_adapter import RealInferencePrefetchAdapter

__all__ = ["SpeculativePrefetcher", "V2Prefetcher", "RealInferencePrefetchAdapter"]
