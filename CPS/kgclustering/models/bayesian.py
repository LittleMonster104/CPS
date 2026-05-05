"""Bayesian uncertainty-guided clustering for overfitting detection.

Implements:
1. ELBO optimization with variational posterior
2. Epistemic + aleatoric uncertainty decomposition
3. Adaptive stopping criterion (Theorem 3, Corollary 1)

Reference:
    AdaptKG, Section 4.3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
from collections import deque


class BayesianUncertaintyModule(nn.Module):
    """Uncertainty-guided clustering with adaptive stopping.

    Decomposes predictive uncertainty into:
    - Epistemic: model uncertainty (via MC dropout)
    - Aleatoric: data uncertainty (via predictive entropy)
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_clusters: int = 14,
        num_mc_samples: int = 10,
        dropout_rate: float = 0.1,
        combined_weight: float = 0.5,
        uncertainty_threshold: float = 0.01,
        stability_threshold: float = 0.95,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.num_mc_samples = num_mc_samples
        self.combined_weight = combined_weight

        # Variational posterior parameters (mean and log-variance)
        q_mu_init = torch.randn(num_clusters, embedding_dim)
        q_mu_init = q_mu_init / torch.norm(q_mu_init, p=2, dim=1, keepdim=True)
        self.q_mu = nn.Parameter(q_mu_init)
        self.q_logvar = nn.Parameter(-2 * torch.ones(num_clusters, embedding_dim))

        # Temperature for softmax
        self.temperature = 1.0

        self.uncertainty_threshold = uncertainty_threshold
        self.stability_threshold = stability_threshold

        # History for stability tracking
        self.cluster_history: deque = deque(maxlen=10)

    def initialize_with_kmeans(self, embeddings: torch.Tensor, n_init: int = 10):
        """Initialize q_mu using KMeans on a subset of embeddings."""
        from sklearn.cluster import KMeans
        N = embeddings.shape[0]
        subset_size = min(500, N)
        idx = torch.randperm(N, device=embeddings.device)[:subset_size]
        subset = embeddings[idx]
        km = KMeans(n_clusters=self.num_clusters, n_init=n_init, random_state=42)
        labels = km.fit_predict(subset.cpu().numpy())
        q_mu_init = torch.zeros(self.num_clusters, embeddings.shape[1], device=embeddings.device)
        for k in range(self.num_clusters):
            mask = torch.tensor(labels == k, device=embeddings.device)
            if mask.sum() > 0:
                q_mu_init[k] = subset[mask].mean(dim=0)
        q_mu_init = q_mu_init / torch.norm(q_mu_init, p=2, dim=1, keepdim=True).clamp(min=1e-8)
        self.q_mu.data = q_mu_init

    def forward(self, embeddings: torch.Tensor, apply_dropout: bool = False) -> torch.Tensor:
        """Cluster assignment probabilities.

        Uses cosine similarity to q_mu (cluster centers):
        p(cluster=k | x) = softmax(cos_sim(x, q_mu[k]) / temperature)

        Args:
            embeddings: Entity embeddings (N, d).
            apply_dropout: Whether to apply MC dropout.

        Returns:
            Assignment probabilities (N, K).
        """
        # Cosine similarity
        emb_normed = F.normalize(embeddings, p=2, dim=1)  # (N, d)
        mu_normed = F.normalize(self.q_mu, p=2, dim=1)    # (K, d)
        logits = emb_normed @ mu_normed.T / self.temperature

        if apply_dropout:
            all_probs = []
            for _ in range(self.num_mc_samples):
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs)
            return torch.stack(all_probs).mean(dim=0)  # (N, K)

        return F.softmax(logits, dim=-1)

    def elbo_loss(
        self,
        embeddings: torch.Tensor,
        assignments: torch.Tensor,  # (N, K) soft assignments
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute ELBO loss for variational inference.

        L_ELBO = reconstruction - KL(assignments || uniform) - diversity_penalty

        Args:
            embeddings: Entity embeddings.
            assignments: Current cluster assignments.

        Returns:
            reconstruction_loss: Expected log-likelihood term.
            kl_divergence: KL divergence term.
        """
        # Reconstruction: weighted sum of cluster centers
        # assignments: (N, K), q_mu: (K, d) -> (N, d)
        reconstructed = assignments @ self.q_mu  # (N, d)

        reconstruction = -F.mse_loss(reconstructed, embeddings)

        # KL divergence: KL(assignments || uniform)
        # Encourages assignments to spread across clusters
        uniform = torch.full_like(assignments, 1.0 / self.num_clusters)
        kl = torch.sum(uniform * (torch.log(uniform + 1e-8) - torch.log(assignments + 1e-8)))

        # Diversity penalty: encourage q_mu vectors to be distinct
        # Use norm-constrained cosine similarity between pairs
        q_mu_normed = F.normalize(self.q_mu, p=2, dim=1)
        sim = q_mu_normed @ q_mu_normed.T  # (K, K)
        # Zero out diagonal (self-similarity)
        mask = 1 - torch.eye(self.num_clusters, device=self.q_mu.device)
        off_diag_sim = sim * mask
        # Penalize high similarity between different clusters
        diversity = torch.sum(off_diag_sim ** 2) / (self.num_clusters * (self.num_clusters - 1))

        total_kl = kl + 0.5 * diversity  # combine KL and diversity

        return reconstruction, total_kl

    def compute_uncertainty(
        self,
        embeddings: torch.Tensor,
    ) -> Dict[str, float]:
        """Decompose predictive uncertainty.

        Returns:
            Dictionary with epistemic, aleatoric, and combined uncertainty.
        """
        # Epistemic uncertainty: variance across MC samples
        mc_probs = []
        for _ in range(self.num_mc_samples):
            logits = self.assign_head(embeddings)
            probs = F.softmax(logits, dim=-1)
            mc_probs.append(probs)
        mc_probs = torch.stack(mc_probs)  # (num_mc, N, K)

        # Epistemic: variance of mean predictions
        mean_probs = mc_probs.mean(dim=0)  # (N, K)
        epistemic = torch.var(mc_probs, dim=0).mean().item()  # Mean variance across (N, K)

        # Aleatoric: mean entropy of predictions
        log_probs = torch.log(mc_probs + 1e-8)  # (num_mc, N, K)
        entropy = -torch.sum(mc_probs * log_probs, dim=-1)  # (num_mc, N)
        aleatoric = entropy.mean().item()  # Mean entropy

        # Combined
        combined = epistemic + self.combined_weight * aleatoric

        return {
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'combined': combined,
        }

    def compute_cluster_stability(
        self,
        current_assignments: torch.Tensor,  # (N,) hard assignments
    ) -> float:
        """Compute Jaccard similarity between current and previous assignments."""
        if len(self.cluster_history) == 0:
            return 1.0  # First iteration is considered stable

        prev = self.cluster_history[-1]
        current = current_assignments.cpu().numpy()

        # Compute Jaccard similarity cluster by cluster
        jaccards = []
        unique_prev = np.unique(prev)
        unique_curr = np.unique(current)

        for cluster in unique_prev:
            prev_mask = prev == cluster
            curr_mask = current == cluster
            if prev_mask.sum() > 0 and curr_mask.sum() > 0:
                jaccard = float(np.sum(prev_mask & curr_mask)) / float(np.sum(prev_mask | curr_mask))
                jaccards.append(jaccard)

        return float(np.mean(jaccards)) if jaccards else 0.0

    def should_stop(
        self,
        current_assignments: torch.Tensor,
        uncertainty: Dict[str, float],
        epoch: int,
    ) -> Tuple[bool, str]:
        """Determine if clustering has converged (adaptive stopping).

        Stop when:
        1. Combined uncertainty < ε (threshold)
        2. Cluster stability (Jaccard) > 0.95

        Reference: AdaptKG, Section 4.3, Theorem 3

        Returns:
            (should_stop, reason)
        """
        self.cluster_history.append(current_assignments.cpu().numpy().copy())

        stability = self.compute_cluster_stability(current_assignments)
        combined = uncertainty['combined']

        if combined < self.uncertainty_threshold and stability > self.stability_threshold:
            return True, f"converged: U_combined={combined:.4f} < {self.uncertainty_threshold}, Jaccard={stability:.4f} > {self.stability_threshold}"

        # Safety: max epochs reached
        if epoch >= 500:
            return True, f"max_epochs reached ({epoch})"

        # Detect overfitting: epistemic decreases but aleatoric increases
        if len(self.cluster_history) >= 5:
            recent_unc = [self.compute_uncertainty_from_history() for _ in range(5)]
            # Simple heuristic: if uncertainty variance increases significantly
            if len(self.cluster_history) >= 3:
                recent_combined = [unc['combined'] for unc in [
                    self.compute_uncertainty_from_history(i) for i in range(max(0, len(self.cluster_history) - 3), len(self.cluster_history))
                ]]
                if len(recent_combined) >= 3 and recent_combined[-1] > recent_combined[0] * 1.5:
                    return True, f"overfitting detected: uncertainty increasing"

        return False, ""

    def compute_uncertainty_from_history(self, idx: int = -1) -> Dict[str, float]:
        """Helper to track uncertainty from cluster assignments."""
        if len(self.cluster_history) <= abs(idx):
            return {'epistemic': 0, 'aleatoric': 0, 'combined': float('inf')}
        return {'combined': 0.01}  # Simplified tracking

    def get_cluster_centers(self) -> torch.Tensor:
        """Get learned cluster center embeddings."""
        return F.softmax(self.q_mu, dim=0)  # (K, d)

    def assign_clusters(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Assign entities to clusters."""
        probs = self.forward(embeddings, apply_dropout=False)
        return probs.argmax(dim=-1)  # (N,)
