"""Knowledge graph dataset loaders for AdaptKG.

Exports:
    KGDataset: Dataclass representing a KG with torch/scipy representation.
    load_dataset: Load by name ('FB15K-237', 'WN18RR', 'YAGO3-10').
    load_fb15k237: Load FB15K-237 dataset.
    load_wn18rr: Load WN18RR dataset.
    load_yago3_10: Load YAGO3-10 dataset.
    load_tsv_triplets: Load TSV triples to numpy array.
    build_adjacency_matrix: Build scipy sparse adjacency from triples.
    compute_degree_stats: Compute degree distribution from sparse adjacency.
"""

from .datasets import (
    KGDataset,
    load_dataset,
    load_fb15k237,
    load_wn18rr,
    load_yago3_10,
    load_tsv_triplets,
    build_adjacency_matrix,
    compute_degree_stats,
    DATASET_STATS,
)

from .sampling import DegreeAwareSampler

__all__ = [
    'KGDataset',
    'load_dataset',
    'load_fb15k237',
    'load_wn18rr',
    'load_yago3_10',
    'load_tsv_triplets',
    'build_adjacency_matrix',
    'compute_degree_stats',
    'DegreeAwareSampler',
    'DATASET_STATS',
]
