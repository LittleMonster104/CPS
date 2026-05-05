"""AdaptKG evaluation package.

Exports:
    compute_nmi: Normalized Mutual Information.
    compute_ari: Adjusted Rand Index.
    compute_hits_at_k: Hits@k for cluster coherence.
    compute_silhouette: Silhouette score.
    evaluate_clustering: All clustering metrics.
"""

from .metrics import (
    compute_nmi,
    compute_ari,
    compute_hits_at_k,
    compute_silhouette,
    evaluate_clustering,
)

__all__ = [
    'compute_nmi',
    'compute_ari',
    'compute_hits_at_k',
    'compute_silhouette',
    'evaluate_clustering',
]
