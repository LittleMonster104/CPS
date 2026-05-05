"""Degree-aware ego-graph sampling (AdaptKG Algorithm 1).

Uses networkx.Graph for the KG and nx.ego_graph for extraction.
Implements stratified sampling with adaptive radius adjustment
to address boundary ambiguity in KG clustering.

Reference:
    AdaptKG, Section 4.1, Theorem 1 (Global Degree Coverage)
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from .datasets import KGDataset


class DegreeAwareSampler:
    """Algorithm 1: Degree-Aware Stratified Sampling.

    Stratifies entities by degree quantiles and applies
    adaptive-radius ego-graph extraction to balance coverage
    across hub and long-tail entities.
    """

    def __init__(
        self,
        target_volume: int = 100,
        quantiles: Tuple[float, float] = (0.25, 0.75),
        min_radius: int = 1,
        max_radius: int = 4,
    ):
        """Initialize the sampler.

        Args:
            target_volume: C - target local volume constant.
            quantiles: (q25, q75) for stratification.
            min_radius: minimum adaptive radius (>= 1).
            max_radius: maximum adaptive radius.
        """
        self.target_volume = target_volume
        self.quantiles = quantiles
        self.min_radius = min_radius
        self.max_radius = max_radius
        self._avg_degree: Optional[float] = None
        self._degree_thresholds: Optional[np.ndarray] = None
        self._graph: Optional[nx.Graph] = None

    def fit(self, dataset: KGDataset) -> 'DegreeAwareSampler':
        """Compute stratification thresholds and build networkx graph from dataset.

        Args:
            dataset: The KG dataset.

        Returns:
            self
        """
        # Build networkx graph from edge_index
        self._graph = nx.Graph()
        for i in range(dataset.entity_count):
            self._graph.add_node(i)
        
        for e in range(dataset.edge_index.shape[0]):
            h = int(dataset.edge_index[e, 0])
            t = int(dataset.edge_index[e, 2])
            r = int(dataset.edge_types[e])
            if h != t:  # avoid self-loops in graph
                if not self._graph.has_edge(h, t):
                    self._graph.add_edge(h, t, relation=r)
                else:
                    # Multi-relational: store as list
                    if 'relations' not in self._graph[h][t]:
                        self._graph[h][t]['relations'] = []
                    self._graph[h][t]['relations'].append(r)

        degrees = np.array([d for n, d in self._graph.degree()], dtype=np.float64)
        self._avg_degree = float(degrees.mean())
        q25, q75 = self.quantiles
        self._degree_thresholds = np.percentile(degrees, [q25 * 100, q75 * 100])
        return self

    def adaptive_radius(self, degree: int) -> int:
        """Compute adaptive radius for an entity given its degree.

        r(e) = max(1, ceil(log_{d_bar}(C . deg(e))))

        High-degree entities get smaller radius, low-degree get larger.

        Args:
            degree: degree of the entity.

        Returns:
            Adaptive radius clamped to [min_radius, max_radius].
        """
        if self._avg_degree is None or self._avg_degree <= 0:
            return self.min_radius
        avg_d = max(self._avg_degree, 2.0)  # avoid log base 1

        radius = int(np.ceil(np.log(self.target_volume / max(degree, 1)) / np.log(avg_d)))
        return int(np.clip(radius, self.min_radius, self.max_radius))

    def stratify(self, dataset: KGDataset) -> Tuple[List[int], List[int], List[int]]:
        """Stratify entities into low/mid/high degree groups.

        Args:
            dataset: The KG dataset.

        Returns:
            (low_degree_indices, mid_degree_indices, high_degree_indices)
        """
        if self._graph is None:
            self.fit(dataset)

        degrees = np.array([d for n, d in self._graph.degree()], dtype=np.float64)
        low_mask = degrees < self._degree_thresholds[0]
        mid_mask = (degrees >= self._degree_thresholds[0]) & (degrees < self._degree_thresholds[1])
        high_mask = degrees >= self._degree_thresholds[1]

        return (
            np.where(low_mask)[0].tolist(),
            np.where(mid_mask)[0].tolist(),
            np.where(high_mask)[0].tolist(),
        )

    def sample_ego_graph(
        self,
        dataset: KGDataset,
        entity: int,
    ) -> Tuple[nx.Graph, np.ndarray, Dict[int, int]]:
        """Extract ego-graph for a single entity with adaptive radius.

        Args:
            dataset: The KG dataset.
            entity: Entity ID to extract ego-graph for.

        Returns:
            ego_subgraph: networkx subgraph containing the ego-network.
            node_map: Mapping from original entity ID to subgraph node index.
            subgraph_nodes: List of original entity IDs in the ego-graph.
        """
        if self._graph is None:
            self.fit(dataset)

        degree = self._graph.degree(entity)
        radius = self.adaptive_radius(degree)
        radius = max(radius, 1)  # at least radius 1

        # Extract ego-graph using networkx
        ego = nx.ego_graph(self._graph, entity, radius=radius)
        
        # Create node mapping
        node_map = {n: i for i, n in enumerate(ego.nodes())}
        subgraph_nodes = list(ego.nodes())
        
        return ego, np.array(list(node_map.values()), dtype=np.int64), subgraph_nodes

    def sample_batch(
        self,
        dataset: KGDataset,
        num_samples: int = 512,
        random_state: Optional[np.random.RandomState] = None,
    ) -> List[Tuple[int, int, List[int]]]:
        """Sample a batch of ego-graphs using stratified sampling.

        Samples from each degree stratum proportionally.

        Args:
            dataset: The KG dataset.
            num_samples: Number of entities to sample.
            random_state: NumPy random state for reproducibility.

        Returns:
            List of (entity_id, radius, subgraph_nodes) tuples.
        """
        if random_state is None:
            random_state = np.random.RandomState(42)

        if self._graph is None:
            self.fit(dataset)

        # Stratify entities
        low, mid, high = self.stratify(dataset)
        
        # Sample proportionally from each stratum
        n_low = max(1, num_samples// 3)
        n_mid = max(1, num_samples// 3)
        n_high = num_samples - n_low - n_mid

        low_sample = random_state.choice(low, size=min(n_low, len(low)), replace=len(low) < n_low)
        mid_sample = random_state.choice(mid, size=min(n_mid, len(mid)), replace=len(mid) < n_mid)
        high_sample = random_state.choice(high, size=min(n_high, len(high)), replace=len(high) < n_high)

        all_entities = np.concatenate([low_sample, mid_sample, high_sample])
        
        results = []
        for entity in all_entities:
            ego, _, nodes = self.sample_ego_graph(dataset, int(entity))
            radius = self.adaptive_radius(self._graph.degree(entity))
            results.append((int(entity), radius, list(nodes)))
        
        return results
