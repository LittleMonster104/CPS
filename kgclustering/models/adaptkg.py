"""Full AdaptKG model integrating all components.

AdaptKG = R-GCN backbone + Relation-Specific Prompting +
          Bayesian Clustering + Domain Adaptation

All components use pure PyTorch - no PyG dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

from .rgcn import RGCNBackbone
from .prompting import RelationSpecificPrompting
from .bayesian import BayesianUncertaintyModule
from .domain_adapt import StructureAwareNormalization, MaximumMeanDiscrepancy


class AdaptKG(nn.Module):
    """Complete AdaptKG model.

    Architecture:
    1. R-GCN backbone: entity embeddings
    2. Relation-specific prompting: semantic separation
    3. Bayesian uncertainty-guided clustering: adaptive stopping
    4. Structure-aware domain adaptation: zero-shot transfer
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 256,
        prompt_length: int = 32,
        temperature: float = 0.5,
        num_clusters: int = 14,
        num_mc_samples: int = 10,
        dropout_rate: float = 0.1,
        mmd_weight: float = 0.1,
        mlp_hidden_dims: Tuple[int, ...] = (64, 64),
    ):
        super().__init__()

        # 1. R-GCN Backbone
        self.backbone = RGCNBackbone(
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            num_relations=num_relations,
            num_layers=3,
            num_bases=5,
            dropout=dropout_rate,
        )
        self.initial_features = None

        # Store num_clusters as instance attribute
        self.num_clusters = num_clusters

        # 2. Relation-Specific Prompting
        self.prompting = RelationSpecificPrompting(
            num_relations=num_relations,
            prompt_dim=embedding_dim,
            prompt_length=prompt_length,
            temperature=temperature,
        )

        # 3. Bayesian Uncertainty-Guided Clustering
        self.bayesian_clustering = BayesianUncertaintyModule(
            embedding_dim=embedding_dim,
            num_clusters=num_clusters,
            num_mc_samples=num_mc_samples,
            dropout_rate=dropout_rate,
        )

        # 4. Domain Adaptation (initialized lazily)
        self.domain_adapt_norm = None
        self.domain_adapt_mmd = MaximumMeanDiscrepancy(lambda_mmd=mmd_weight)
        self.mmd_weight = mmd_weight

        # Loss weights
        self.loss_weights = {
            "semantic": 1.0,
            "cluster": 1.0,
            "mmd": mmd_weight,
        }

    def initialize(self, num_entities: int):
        """Initialize learnable features."""
        if self.initial_features is None:
            self.initial_features = self.backbone.get_initial_features(num_entities)

    def forward(
        self,
        edge_index: torch.Tensor,  # (E, 3): [head, relation, tail]
        edge_types: torch.Tensor,  # (E,) relation type for each edge
        relation_indices: torch.Tensor,  # (batch,) which relations are active
        apply_uncertainty: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass through AdaptKG.

        Args:
            edge_index: (E, 3) edge list for R-GCN.
            edge_types: (E,) relation type for each edge.
            relation_indices: (batch,) which relations are active.
            apply_uncertainty: Whether to use MC dropout for uncertainty.

        Returns:
            Dict with keys: embeddings, cluster_probs, semantic_loss, etc.
        """
        # Step 1: Get entity embeddings from R-GCN backbone
        if self.initial_features is not None:
            x = self.initial_features
        else:
            x = torch.zeros(edge_index[:, 0].max().item() + 1, self.backbone.hidden_dim,
                          device=edge_index.device)
        
        embeddings = self.backbone(x, edge_index, edge_types)

        # Step 2: Relation-specific prompting
        # Compute per-batch global embedding (mean over entities of same relation)
        # Get entities connected by each relation
        unique_rels = torch.unique(relation_indices)
        global_embs_for_rels = {}
        for rel in unique_rels:
            rel_id = int(rel)
            # Find entities (both head and tail) of this relation
            rel_edges = torch.where(edge_types == rel_id)[0]
            if len(rel_edges) > 0:
                connected = torch.cat([edge_index[rel_edges, 0], edge_index[rel_edges, 2]])
                unique_entities = connected.unique()
                global_embs_for_rels[rel_id] = embeddings[unique_entities].mean(dim=0)
        
        # Build per-batch global embeddings
        rel_idx_list = relation_indices.tolist()
        batch_global = torch.zeros(len(rel_idx_list), self.backbone.hidden_dim, device=edge_index.device)
        for i, rel_idx in enumerate(rel_idx_list):
            rel_id = int(rel_idx)
            if rel_id in global_embs_for_rels:
                batch_global[i] = global_embs_for_rels[rel_id]
            else:
                batch_global[i] = embeddings.mean(dim=0)

        prompt_output = self.prompting(
            local_embeddings=torch.zeros(1, 1, self.prompting.prompt_dim, device=embeddings.device),
            relation_indices=relation_indices,
            global_embedding=batch_global,  # (batch, hidden_dim)
        )
        relation_emb = prompt_output[0]

        # Step 3: Cluster assignments
        cluster_probs = self.bayesian_clustering(embeddings, apply_dropout=apply_uncertainty)

        return {
            "embeddings": embeddings,
            "relation_embeddings": relation_emb,
            "cluster_probs": cluster_probs,
            "loss_weights": self.loss_weights,
        }

    def enable_domain_adaptation(
        self,
        prompt_dim: int = 256,
        hidden_dims: Tuple[int, ...] = (64, 64),
        num_structural_features: int = 3,
    ):
        """Enable domain adaptation by initializing the normalization module."""
        self.domain_adapt_norm = StructureAwareNormalization(
            prompt_dim=prompt_dim,
            hidden_dims=hidden_dims,
            num_structural_features=num_structural_features,
        )

    def compute_structural_stats(self, dataset) -> torch.Tensor:
        """Compute structural statistics from a KG dataset."""
        adj = dataset.adj_sparse
        N = dataset.entity_count
        E = dataset.edge_index.shape[0]

        sparsity = 1.0 - E / (N * N) if N > 0 else 0.0
        density = float(dataset.avg_degree)

        degrees = np.array(adj.sum(axis=1)).flatten().astype(np.float64) + 1e-8
        sorted_degrees = np.sort(degrees)[::-1][:min(1000, len(degrees))]
        log_deg = np.log(sorted_degrees)
        log_rank = np.log(np.arange(1, len(log_deg) + 1))
        if len(log_rank) > 2:
            slope = np.polyfit(log_rank, log_deg, 1)[0]
            cluster_shape = abs(slope)
        else:
            cluster_shape = 1.0

        return torch.tensor([sparsity, density, cluster_shape], dtype=torch.float32)

    def assign_clusters(
        self,
        embeddings: torch.Tensor,
        apply_dropout: bool = False,
    ) -> torch.Tensor:
        """Assign each entity to its most likely cluster.

        Args:
            embeddings: Entity embeddings (N, d).
            apply_dropout: Whether to apply MC dropout.

        Returns:
            Cluster assignments (N,) with integer labels.
        """
        probs = self.bayesian_clustering(embeddings, apply_dropout=apply_dropout)
        return torch.argmax(probs, dim=1)

    def semantic_separation_loss(
        self,
        relation_embeddings: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Compute semantic separation loss between different relation embeddings.

        Encourages different relation embeddings to be semantically distinct.

        Args:
            relation_embeddings: Dict mapping relation_id -> embedding tensor.

        Returns:
            Scalar loss value.
        """
        if len(relation_embeddings) < 2:
            return torch.tensor(0.0, device=next(iter(relation_embeddings.values())).device)

        embs = torch.stack(list(relation_embeddings.values()))  # (num_rels, d)
        # Normalize
        norms = embs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        embs = embs / norms

        # Compute pairwise cosine similarity
        sim = torch.mm(embs, embs.T)  # (num_rels, num_rels)
        # Loss: encourage dissimilarity (diagonal is always 1, so mask it)
        mask = torch.eye(sim.size(0), device=sim.device) == 0
        return -sim[mask].mean()

    def mmd_loss(
        self,
        source_emb: torch.Tensor,
        target_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MMD between source and target embeddings."""
        return self.domain_adapt_mmd(source_emb, target_emb)

    def get_relation_embeddings(
        self,
        embeddings: torch.Tensor,
        relation_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Get relation-conditioned embeddings.

        Args:
            embeddings: Entity embeddings (N, d).
            relation_indices: Which relations are active (batch,).

        Returns:
            Relation-conditioned embeddings (batch, d).
        """
        return self.prompting(embeddings, relation_indices)[0]

    def save_source_state(
        self,
        prompts,
        embeddings,
    ):
        """Save source-domain state for transfer.

        Stores in module attributes so they can be loaded during transfer.
        """
        self._source_prompts = prompts
        self._source_embeddings = embeddings

    def forward_with_domain_adapt(
        self,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        relation_indices: torch.Tensor,
        target_structural_stats: torch.Tensor,
        apply_uncertainty: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with domain adaptation enabled.

        Args:
            edge_index: (E, 3) edge list.
            edge_types: (E,) relation types.
            relation_indices: Active relations.
            target_structural_stats: Target domain structural features.
            apply_uncertainty: Use MC dropout.

        Returns:
            Dict with embeddings, cluster_probs, and adaptation info.
        """
        # Get base embeddings
        base = self.forward(edge_index, edge_types, relation_indices, apply_uncertainty)

        if self.domain_adapt_norm is not None and hasattr(self, "_source_prompts"):
            adapted = self.domain_adapt_norm(
                self._source_prompts,
                target_structural_stats.unsqueeze(0),
            )
            base["adapted_embeddings"] = adapted.mean(dim=1) if adapted.dim() > 2 else adapted

        return base
