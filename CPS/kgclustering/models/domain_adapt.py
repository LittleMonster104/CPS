"""Structure-aware domain adaptation for zero-shot cross-domain transfer.

Implements:
1. Structure-aware normalization (MLP-adapted prompts)
2. MMD distribution alignment
3. Zero-shot transfer guarantee (Theorem 4, Corollary 2)

Reference:
    AdaptKG, Section 4.4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class StructureAwareNormalization(nn.Module):
    """MLP that adapts source-domain prompts to target-domain statistics.

    Prompt_t = γ · Prompt_s + β
    where γ, β = MLP([sparsity, density, cluster_shape])

    For sparse domains: aggressive scaling (γ > 1)
    For dense domains: conservative scaling (γ < 1)
    """

    def __init__(
        self,
        prompt_dim: int = 256,
        hidden_dims: Tuple[int, ...] = (64, 64),
        num_structural_features: int = 3,
    ):
        super().__init__()
        layers = []
        in_dim = num_structural_features
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 2 * prompt_dim))  # γ and β
        self.mlp = nn.Sequential(*layers)

    def forward(self, source_prompts: torch.Tensor, structural_stats: torch.Tensor) -> torch.Tensor:
        """Adapt source prompts to target domain.

        Args:
            source_prompts: Source-domain prompts (M, L, d).
            structural_stats: Target graph statistics (batch, 3) = [sparsity, density, cluster_shape].

        Returns:
            Adapted prompts (M, L, d).
        """
        params = self.mlp(structural_stats)  # (batch, 2*d)
        gamma = params[:, :self.mlp[-1].out_features // 2].unsqueeze(0)  # (1, M, d)
        beta = params[:, self.mlp[-1].out_features // 2:].unsqueeze(0)

        # Apply affine transformation
        adapted = gamma * source_prompts + beta
        return adapted

    def compute_structural_stats(self, dataset) -> torch.Tensor:
        """Compute structural statistics from a KG dataset.

        Returns tensor [sparsity, density, cluster_shape].

        Args:
            dataset: KGDataset with adj_sparse, entity_count, edge_index, avg_degree.
        """
        adj = dataset.adj_sparse
        N = dataset.entity_count
        E = dataset.edge_index.shape[0]

        # Sparsity: 1 - |E| / |V|^2
        sparsity = 1.0 - E / (N * N) if N > 0 else 0.0

        # Density: average degree
        density = float(dataset.avg_degree)

        # Cluster shape: power-law exponent of degree distribution
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


class MaximumMeanDiscrepancy(nn.Module):
    """Compute MMD between source and target distributions.

    L_MMD = ||μ_s - μ_t||^2 + λ ||Σ_s - Σ_t||_F^2
    """

    def __init__(self, lambda_mmd: float = 0.1):
        super().__init__()
        self.lambda_mmd = lambda_mmd

    def forward(
        self,
        source_embeddings: torch.Tensor,  # (N_s, d)
        target_embeddings: torch.Tensor,  # (N_t, d)
    ) -> torch.Tensor:
        """Compute MMD loss.

        Args:
            source_embeddings: Source domain embeddings.
            target_embeddings: Target domain embeddings.

        Returns:
            MMD loss scalar.
        """
        # First moment alignment
        mu_s = source_embeddings.mean(dim=0)
        mu_t = target_embeddings.mean(dim=0)
        moment1_loss = torch.norm(mu_s - mu_t) ** 2

        # Second moment alignment (covariance)
        cov_s = torch.cov(source_embeddings.T)
        cov_t = torch.cov(target_embeddings.T)
        moment2_loss = torch.norm(cov_s - cov_t, p='fro') ** 2

        return moment1_loss + self.lambda_mmd * moment2_loss


class DomainAdaptationModule:
    """Full domain adaptation pipeline for zero-shot transfer.

    Combines structure-aware normalization and MMD alignment
    to enable robust transfer without target-domain fine-tuning.
    """

    def __init__(
        self,
        prompt_dim: int = 256,
        hidden_dims: Tuple[int, ...] = (64, 64),
        mmd_weight: float = 0.1,
    ):
        self.normalization = StructureAwareNormalization(
            prompt_dim=prompt_dim,
            hidden_dims=hidden_dims,
        )
        self.mmd = MaximumMeanDiscrepancy(lambda_mmd=mmd_weight)

    def fit_source(
        self,
        source_prompts: Dict[int, torch.Tensor],
        source_embeddings: torch.Tensor,
    ):
        """Store source-domain learned prompts and embeddings."""
        self.source_prompts = source_prompts
        self.source_embeddings = source_embeddings

    def transfer(
        self,
        target_structural_stats: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """Transfer prompts to target domain.

        Args:
            target_structural_stats: Structural statistics of target KG.

        Returns:
            Adapted prompts for target domain.
        """
        adapted = {}
        source_list = list(self.source_prompts.values())
        source_tensor = torch.stack(source_list)  # (M, L, d)

        adapted_tensor = self.normalization(source_tensor, target_structural_stats)

        for r_idx, prompt in enumerate(adapted_tensor):
            adapted[r_idx] = prompt

        return adapted

    def compute_transfer_error_bound(
        self,
        source_error: float,
        source_embs: torch.Tensor,
        target_embs: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute domain adaptation bound: ε_t ≤ ε_s + D_MMD + λ*.

        Args:
            source_error: Source domain clustering error.
            source_embs: Source embeddings.
            target_embs: Target embeddings.

        Returns:
            Dictionary with error bound components.
        """
        mmd_loss = self.mmd(source_embs, target_embs).item()
        n = min(len(source_embs), len(target_embs))
        delta = 0.05
        estimation_error = np.sqrt(np.log(1 / delta) / n)

        upper_bound = source_error + mmd_loss + estimation_error

        return {
            'source_error': source_error,
            'mmd_distance': mmd_loss,
            'estimation_error': estimation_error,
            'upper_bound': upper_bound,
        }
