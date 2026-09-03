"""Flash Translation Layer (FTL) implementations: Conventional & Tensor-Aware."""
from person2_ssd.ftl.base import BaseFTL
from person2_ssd.ftl.mapping import MappingTable
from person2_ssd.ftl.conventional import ConventionalFTL
from person2_ssd.ftl.tensor_aware import TensorAwareFTL

__all__ = ["BaseFTL", "MappingTable", "ConventionalFTL", "TensorAwareFTL"]
