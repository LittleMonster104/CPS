"""Load FB15K-237, WN18RR, YAGO3-10, PrimeKG datasets for AdaptKG.

Pure PyTorch + scipy + numpy implementation. No PyG dependencies.

Data format:
- Standard FB15K-237 TSV: head_id, relation_id, tail_id (integer-encoded)
- Adjacency as scipy.sparse.coo_matrix
- Edge list as torch long tensor (E x 3)
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class KGDataset:
    """Knowledge graph dataset for AdaptKG (pure PyTorch/scipy representation).

    Attributes:
        name: Dataset name (e.g., 'FB15K-237').
        entity_count: Total number of entities (N).
        relation_count: Total number of relations (M).
        edge_index: Edge list tensor of shape (E, 3), each row = [head, relation, tail].
        edge_types: Relation type tensor of shape (E,).
        adj_sparse: Scipy sparse adjacency matrix of shape (N, N).
        entity_types: Dict mapping entity_id -> ground-truth cluster_label.
        num_clusters: Number of ground-truth clusters (for evaluation).
        avg_degree: Average degree of the graph.
        structure_type: One of 'dense', 'sparse', 'hierarchical', 'mixed'.
    """
    name: str
    entity_count: int
    relation_count: int
    edge_index: torch.Tensor  # (E, 3)
    edge_types: torch.Tensor  # (E,)
    adj_sparse: sp.csr_matrix
    entity_types: Dict[int, str] = field(default_factory=dict)
    num_clusters: int = 0
    avg_degree: float = 0.0
    structure_type: str = 'dense'

    def __post_init__(self):
        assert self.edge_index.shape[0] == self.edge_types.shape[0]
        assert self.edge_index.shape[1] == 3


# Dataset statistics for reference
DATASET_STATS = {
    'FB15K-237': {
        'num_entities': 14541,
        'num_relations': 237,
        'avg_degree': 18.4,
        'structure_type': 'dense',
        'num_clusters': 14,
    },
    'WN18RR': {
        'num_entities': 40943,
        'num_relations': 11,
        'avg_degree': 2.6,
        'structure_type': 'sparse',
        'num_clusters': 11,
    },
    'YAGO3-10': {
        'num_entities': 123182,
        'num_relations': 37,
        'avg_degree': 12.3,
        'structure_type': 'hierarchical',
        'num_clusters': 10,
    },
    'PrimeKG': {
        'num_entities': 129375,
        'num_relations': 29,
        'avg_degree': 31.4,
        'structure_type': 'mixed',
        'num_clusters': 8,
    },
}


def load_tsv_triplets(filepath: str) -> np.ndarray:
    """Load a TSV file of triples (head_id, relation_id, tail_id).

    Args:
        filepath: Path to TSV file.

    Returns:
        triples: numpy array of shape (E, 3) with integer-encoded triples.
    """
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(chr(9))
            if len(parts) == 3:
                rows.append([int(parts[0]), int(parts[1]), int(parts[2])])
    return np.array(rows, dtype=np.int64)


def build_adjacency_matrix(triples: np.ndarray, num_entities: int) -> sp.csr_matrix:
    """Build sparse adjacency matrix from triples.

    Creates an undirected adjacency matrix: for each (h, r, t),
    adds edges (h, t) and (t, h) with weight 1.

    Args:
        triples: Array of shape (E, 3) with [head, relation, tail].
        num_entities: Number of entities N.

    Returns:
        Sparse CSR matrix of shape (N, N).
    """
    rows = np.concatenate([triples[:, 0], triples[:, 2]])
    cols = np.concatenate([triples[:, 2], triples[:, 0]])
    data = np.ones(len(rows), dtype=np.float32)

    adj = sp.coo_matrix((data, (rows, cols)), shape=(num_entities, num_entities))
    adj.eliminate_zeros()
    return adj.tocsr()


def compute_degree_stats(adj: sp.csr_matrix) -> Tuple[np.ndarray, float]:
    """Compute degree distribution and average degree from sparse adjacency.

    Args:
        adj: Sparse CSR matrix of shape (N, N).

    Returns:
        degrees: Array of N degrees.
        avg_degree: Mean degree across all entities.
    """
    row_sums = np.array(adj.sum(axis=1)).flatten()
    degrees = row_sums.astype(np.float64)
    avg_degree = float(degrees.mean())
    return degrees, avg_degree


def load_fb15k237(data_dir: str = '../data') -> KGDataset:
    """Load FB15K-237 dataset from local files or generate synthetic.

    Looks for train.txt, valid.txt, test.txt in data_dir.
    If not found, generates a small synthetic dataset for testing.

    Args:
        data_dir: Directory containing FB15K-237 data files.

    Returns:
        KGDataset with proper edge tensors and sparse adjacency.
    """
    stats = DATASET_STATS['FB15K-237']

    # Try to load from files
    train_path = os.path.join(data_dir, 'FB15K-237', 'train.txt')
    if not os.path.exists(train_path):
        # Try alternate directory structure
        train_path = os.path.join(data_dir, 'train.txt')

    if os.path.exists(train_path):
        return _load_from_files(train_path, 'FB15K-237', stats)

    # Generate synthetic dataset for testing
    print(f'[load_fb15k237] No data found at {data_dir}, generating synthetic dataset')
    return _generate_synthetic('FB15K-237', stats)


def load_wn18rr(data_dir: str = '../data') -> KGDataset:
    """Load WN18RR dataset."""
    stats = DATASET_STATS['WN18RR']
    train_path = os.path.join(data_dir, 'WN18RR', 'train.txt')
    if not os.path.exists(train_path):
        train_path = os.path.join(data_dir, 'train.txt')

    if os.path.exists(train_path):
        return _load_from_files(train_path, 'WN18RR', stats)

    return _generate_synthetic('WN18RR', stats)


def load_yago3_10(data_dir: str = '../data') -> KGDataset:
    """Load YAGO3-10 dataset."""
    stats = DATASET_STATS['YAGO3-10']
    train_path = os.path.join(data_dir, 'YAGO3-10', 'train.txt')
    if not os.path.exists(train_path):
        train_path = os.path.join(data_dir, 'train.txt')

    if os.path.exists(train_path):
        return _load_from_files(train_path, 'YAGO3-10', stats)

    return _generate_synthetic('YAGO3-10', stats)


def load_dataset(name: str, data_dir: str = '../data') -> KGDataset:
    """Load a dataset by name.

    Args:
        name: Dataset name ('FB15K-237', 'WN18RR', 'YAGO3-10', 'PrimeKG').
        data_dir: Directory containing dataset files.

    Returns:
        KGDataset object.
    """
    loaders = {
        'FB15K-237': load_fb15k237,
        'WN18RR': load_wn18rr,
        'YAGO3-10': load_yago3_10,
    }
    loader = loaders.get(name)
    if loader is None:
        raise ValueError(f'Unknown dataset: {name}. Available: {list(loaders.keys())}')
    return loader(data_dir)


def _load_from_files(filepath: str, name: str, stats: dict) -> KGDataset:
    """Load dataset from TSV files.

    Args:
        filepath: Path to train TSV file.
        name: Dataset name.
        stats: Dataset statistics dict.

    Returns:
        KGDataset object.
    """
    triples = load_tsv_triplets(filepath)
    num_entities = stats['num_entities']
    num_relations = stats['num_relations']

    edge_index = torch.tensor(triples, dtype=torch.long)
    edge_types = torch.tensor(triples[:, 1], dtype=torch.long)

    adj = build_adjacency_matrix(triples, num_entities)
    degrees, avg_degree = compute_degree_stats(adj)

    return KGDataset(
        name=name,
        entity_count=num_entities,
        relation_count=num_relations,
        edge_index=edge_index,
        edge_types=edge_types,
        adj_sparse=adj,
        entity_types={},
        num_clusters=stats['num_clusters'],
        avg_degree=avg_degree,
        structure_type=stats['structure_type'],
    )


def _generate_synthetic(name: str, stats: dict) -> KGDataset:
    """Generate a small synthetic knowledge graph for testing.

    Creates a graph with community structure suitable for clustering evaluation.
    Scales down from full stats to keep the graph manageable for testing.

    Args:
        name: Dataset name.
        stats: Dataset statistics dict.

    Returns:
        KGDataset with synthetic data.
    """
    np.random.seed(42)

    # Scale down for testing: max 200 entities, 5 relations, 10 clusters
    n_entities = min(stats['num_entities'], 200)
    n_relations = min(stats['num_relations'], 5)
    num_clusters = min(stats['num_clusters'], 10)

    # Generate community structure
    entities_per_cluster = n_entities// num_clusters
    entity_types = {}
    for cluster_id in range(num_clusters):
        start = cluster_id * entities_per_cluster
        end = start + entities_per_cluster
        for i in range(start, min(end, n_entities)):
            entity_types[i] = str(cluster_id)

    # Within-cluster edges (dense)
    edges = []
    for cluster_id in range(num_clusters):
        start = cluster_id * entities_per_cluster
        end = min(start + entities_per_cluster, n_entities)
        cluster_size = end - start
        rel_id = cluster_id % n_relations

        # Dense within-cluster edges
        n_edges = max(cluster_size * 5, cluster_size * (cluster_size - 1)// 4)
        for _ in range(n_edges):
            h = np.random.randint(start, end)
            t = np.random.randint(start, end)
            if h != t:
                edges.append([h, rel_id, t])

        # Sparse between-cluster edges
        next_cluster = (cluster_id + 1) % num_clusters
        other_start = next_cluster * entities_per_cluster
        other_end = min(other_start + entities_per_cluster, n_entities)
        if other_start < other_end and other_start < n_entities:
            for _ in range(max(cluster_size, 5)):
                h = np.random.randint(start, end)
                t = np.random.randint(other_start, other_end)
                if h < n_entities and t < n_entities and h != t:
                    edges.append([h, (rel_id + 1) % n_relations, t])

    if not edges:
        edges = [[0, 0, 1], [1, 0, 0]]

    triples = np.array(edges, dtype=np.int64)
    max_entity = triples.max() + 1
    entity_count = max(n_entities, max_entity)

    edge_index = torch.tensor(triples, dtype=torch.long)
    edge_types = torch.tensor(triples[:, 1], dtype=torch.long)

    adj = build_adjacency_matrix(triples, entity_count)
    degrees, avg_degree = compute_degree_stats(adj)

    return KGDataset(
        name=name,
        entity_count=entity_count,
        relation_count=n_relations,
        edge_index=edge_index,
        edge_types=edge_types,
        adj_sparse=adj,
        entity_types=entity_types,
        num_clusters=num_clusters,
        avg_degree=avg_degree,
        structure_type=stats['structure_type'],
    )
