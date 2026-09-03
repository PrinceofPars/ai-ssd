"""
End-to-End Simulation Pipeline: Executes multi-layer decoding inference loop.
"""

from typing import Dict, Any, List
from person3_system.integration.orchestrator import SystemOrchestrator
from person3_system.api.requests import RequestFactory


class InferencePipeline:
    def __init__(self, orchestrator: SystemOrchestrator):
        self.orchestrator = orchestrator

    def run_decode_layer(
        self,
        layer_id: int,
        head_id: int,
        candidate_blocks: List[int],
        top_k: int = 16,
    ) -> Dict[str, Any]:
        req = RequestFactory.create_topk_request(
            layer_id=layer_id,
            head_id=head_id,
            candidate_blocks=candidate_blocks,
            top_k=top_k,
        )
        resp = self.orchestrator.api.execute(req)
        return resp.to_dict()
