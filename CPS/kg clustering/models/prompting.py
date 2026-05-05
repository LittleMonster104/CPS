"""Relation-specific prompting with contrastive semantic separation.

Implements:
1. Per-relation prompt vectors P_r ∈ R^(L×d)
2. Semantic separation loss (Eq. 1 in AdaptKG)
3. Multi-scale aggregation (Eq. 2 in AdaptKG)

Reference:
    AdaptKG, Section 4.2, Theorem 2 (Collapse Prevention Guarantee)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class RelationSpecificPrompting(nn.Module):
    """Relation-specific prompt vectors with contrastive semantic separation.

    Learns per-relation prompt vectors P_r ∈ R^(L×d) and combines
    local relation-specific embeddings with global context via
    learnable gating (α).
    """

    def __init__(
        self,
        num_relations: int,
        prompt_dim: int = 256,
        prompt_length: int = 32,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.num_relations = num_relations
        self.prompt_dim = prompt_dim
        self.prompt_length = prompt_length
        self.temperature = temperature

        # Learnable per-relation prompt vectors
        self.prompts = nn.ParameterDict({
            f"P_{r}": nn.Parameter(torch.randn(prompt_length, prompt_dim) * 0.02)
            for r in range(num_relations)
        })

        # Multi-scale aggregation gate
        self.gate = nn.Parameter(torch.tensor(0.5))

        # Relation embedding projection
        self.proj = nn.Linear(prompt_dim, prompt_dim)

    def forward(
        self,
        local_embeddings: torch.Tensor,
        relation_indices: torch.Tensor,
        global_embedding: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply relation-specific prompts with multi-scale aggregation.

        For each entity, retrieves its relation-specific prompt, applies
        gated projection to the local embedding, and combines with the
        global embedding via attention.

        Args:
            local_embeddings: Local node embeddings from the R-GCN backbone.
            relation_indices: Relation type for each sample.
            global_embedding: Global graph embedding (optional).

        Returns:
            relation_emb: Relation-specific embeddings.
            aggregated: Aggregated embeddings after multi-scale fusion.
        """
        # Get local embeddings
        if local_embeddings.ndim == 2:
            local_embeddings = local_embeddings.unsqueeze(0)  # (1, E, d)
        
        batch_size = relation_indices.size(0)
        
        # Project local embeddings
        local_proj = self.proj(local_embeddings)  # (1, E, L*d) -> project
        
        # Get per-relation prompt features
        prompt_features = []
        unique_rels = relation_indices.unique()
        
        for r_idx in unique_rels:
            if r_idx in self.prompts:
                prompt = self.prompts[f"P_{r_idx}"]  # (L, d)
                # Get embeddings for this relation
                mask = (relation_indices == r_idx)
                idx = torch.where(mask)[0]
                if len(idx) > 0:
                    local = local_proj[0][idx]  # (L, d)
                    prompted = local * F.sigmoid(prompt)
                    prompt_features.append((r_idx, prompted))

        if prompt_features:
            # Mean across samples for each relation
            relation_embs = {}
            for r_idx, prompted in prompt_features:
                relation_embs[int(r_idx)] = prompted.mean(dim=0)
            
            # Build per-sample relation embeddings
            relation_emb_list = []
            for rel_idx in relation_indices:
                r = int(rel_idx)
                if r in relation_embs:
                    relation_emb_list.append(relation_embs[r])
                else:
                    relation_emb_list.append(torch.zeros(self.prompt_dim, device=local_embeddings.device))
            relation_emb = torch.stack(relation_emb_list)  # (batch_size, d)
        else:
            relation_emb = torch.zeros(batch_size, self.prompt_dim,
                                       device=local_embeddings.device)

        # Multi-scale aggregation (Eq. 2)
        alpha = torch.sigmoid(self.gate)
        if global_embedding is not None:
            # Ensure shapes match: both should be (batch_size, d)
            if global_embedding.ndim == 1:
                global_embedding = global_embedding.unsqueeze(0)  # (1, d)
            if global_embedding.size(0) == 1:
                global_embedding = global_embedding[:batch_size]
            aggregated = alpha * relation_emb + (1 - alpha) * global_embedding
        else:
            aggregated = relation_emb

        return relation_emb, aggregated

    def semantic_separation_loss(
        self,
        relation_embeddings: Dict[int, torch.Tensor],  # {r: (batch_r, d)}
        relation_indices: torch.Tensor,  # (batch,)
        target_embeddings: torch.Tensor,  # (batch, d) - same-relation positives
    ) -> torch.Tensor:
        """Contrastive semantic separation loss (Eq. 1 in AdaptKG).

        L_semantic = -log[exp(sim(h_e^{r1}, h_{e'}^{r1})/tau) / sum_r exp(sim(h_e^r, h_{e'}^r)/tau)]

        Maximizes similarity for same-relation pairs while pushing apart
        different relations.

        Args:
            relation_embeddings: Per-relation embeddings.
            relation_indices: Relation index for each sample.
            target_embeddings: Positive pair embeddings (same relation).

        Returns:
            Semantic separation loss scalar.
        """
        if isinstance(relation_indices, list):
            relation_indices = torch.tensor(relation_indices, dtype=torch.long)
        batch_size = relation_indices.size(0)

        # Compute similarity matrix between all relation embeddings
        sims = []
        for r, emb in relation_embeddings.items():
            # Normalize for cosine similarity
            emb_norm = F.normalize(emb, dim=-1)
            tgt_norm = F.normalize(target_embeddings, dim=-1)
            sim = (emb_norm @ tgt_norm.T) / self.temperature
            sims.append(sim.unsqueeze(0))  # (1, batch_r, batch)

        if not sims:
            return torch.tensor(0.0, device=target_embeddings.device)

        # Stack all relation similarities
        all_sims = torch.cat(sims, dim=0)  # (num_rels, batch, batch)

        # For each sample, find the positive relation similarity
        pos_sim = torch.zeros(batch_size, device=target_embeddings.device)
        for i, r in enumerate(relation_indices):
            r_int = int(r)
            if r_int in relation_embeddings:
                r_idx = list(relation_embeddings.keys()).index(r_int)
                pos_sim[i] = all_sims[r_idx, i, i]

        # Compute log-sum-exp for denominator (numerically stable)
        log_sum_exp = torch.logsumexp(all_sims, dim=0)  # (batch,)

        # Contrastive loss
        loss = -pos_sim + log_sum_exp
        return loss.mean()

    def get_relation_embeddings(
        self,
        entity_embeddings: torch.Tensor,  # (num_entities, hidden_dim)
    ) -> Dict[int, torch.Tensor]:
        """Get relation-specific embeddings for all entities.

        Args:
            entity_embeddings: GNN output embeddings.

        Returns:
            Dictionary mapping relation ID to entity embeddings under that relation.
        """
        relation_embs = {}
        for r_idx in range(self.num_relations):
            key = f"P_{r_idx}"
            if key in self.prompts:
                # Apply relation-specific prompt
                prompt = self.prompts[key]
                projected = self.proj(entity_embeddings)
                relation_embs[r_idx] = projected * F.sigmoid(prompt)
        return relation_embs

    def compute_semantic_separation(
        self,
        relation_embs: Dict[int, torch.Tensor],
        entity_rels: Dict[int, int],  # entity -> primary relation
    ) -> torch.Tensor:
        """Compute pairwise semantic separation between relations.

        Delta_{r1,r2}(e_i, e_j) = ||h_{e_i}^{r1} - h_{e_j}^{r1}|| - ||h_{e_i}^{r2} - h_{e_j}^{r2}||

        Returns:
            Mean semantic separation across all relation pairs.
        """
        sorted_rels = sorted(relation_embs.keys())
        separations = []

        for i in range(len(sorted_rels)):
            for j in range(i + 1, len(sorted_rels)):
                r1, r2 = sorted_rels[i], sorted_rels[j]
                emb1 = F.normalize(relation_embs[r1], dim=-1)
                emb2 = F.normalize(relation_embs[r2], dim=-1)

                # Pairwise distances
                dist1 = torch.cdist(emb1, emb1, p=2)
                dist2 = torch.cdist(emb2, emb2, p=2)

                # Mean separation
                mask = dist1 > 0  # exclude self-pairs
                separation = (dist1[mask] - dist2[mask]).mean()
                separations.append(separation)

        if separations:
            return torch.stack(separations).mean()
        return torch.tensor(0.0)
