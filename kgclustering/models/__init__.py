"""AdaptKG models package.

Exports:
    RGCNBackbone: 3-layer R-GCN with basis decomposition (pure PyTorch).
    AdaptKG: Full AdaptKG model.
    RelationSpecificPrompting: Relation-specific prompt vectors.
    BayesianUncertaintyModule: Bayesian uncertainty-guided clustering.
    StructureAwareNormalization: Structure-aware prompt normalization.
    MaximumMeanDiscrepancy: MMD distribution alignment.
"""

from .rgcn import RGCNBackbone
from .adaptkg import AdaptKG
from .prompting import RelationSpecificPrompting
from .bayesian import BayesianUncertaintyModule
from .domain_adapt import StructureAwareNormalization, MaximumMeanDiscrepancy

__all__ = [
    'RGCNBackbone',
    'AdaptKG',
    'RelationSpecificPrompting',
    'BayesianUncertaintyModule',
    'StructureAwareNormalization',
    'MaximumMeanDiscrepancy',
]
