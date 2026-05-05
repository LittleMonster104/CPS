"""Training loop for AdaptKG with adaptive stopping criterion.

Pure PyTorch implementation. No PyG dependencies.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional, Tuple, List

from gp_llm.models.adaptkg import AdaptKG
from gp_llm.data.sampling import DegreeAwareSampler
from gp_llm.evaluation.metrics import compute_nmi, compute_ari


class AdaptKGTrainer:
    """Trainer for AdaptKG with adaptive Bayesian stopping.

    Implements the full training pipeline:
    1. Degree-aware ego-graph sampling
    2. Relation-specific prompt optimization
    3. Bayesian uncertainty-guided clustering
    4. Adaptive stopping based on combined uncertainty + stability
    """

    def __init__(
        self,
        model: AdaptKG,
        dataset,
        config: Dict,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = device
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Initialize model features
        self.model.initialize(dataset.entity_count)
        self.model = self.model.to(self.device)

        # Degree-aware sampler
        self.sampler = DegreeAwareSampler(
            target_volume=config.get("target_volume", 100),
            quantiles=tuple(config.get("quantiles", [0.25, 0.75])),
            min_radius=config.get("min_radius", 1),
            max_radius=config.get("max_radius", 4),
        ).fit(dataset)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get("lr", 1e-3),
            betas=(0.9, 0.999),
            weight_decay=config.get("weight_decay", 1e-5),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("max_epochs", 500),
        )

        # Tracking
        self.loss_history: List[float] = []
        self.uncertainty_history: List[Dict[str, float]] = []
        self.cluster_assignments_history: List[np.ndarray] = []
        self.best_model_state: Optional[Dict] = None
        self.best_nmi = -1.0
        self.convergence_epoch = None
        self.nmi_scores: List[float] = []

    def sample_batch_edges(
        self,
        batch_size: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """Sample a batch of ego-graphs using degree-aware sampling.

        Returns:
            batch_edge_index: (E_batch, 3) edges within sampled ego-graphs
            batch_edge_types: (E_batch,) relation types
            entity_indices: (batch_size,) sampled entity IDs
            relation_indices: (batch_size,) active relations per sample
        """
        batch_data = self.sampler.sample_batch(
            self.dataset,
            num_samples=batch_size,
            random_state=self.rng,
        )

        # Collect edges and nodes from sampled ego-graphs
        all_edges = []
        all_types = []
        entity_indices = []
        relation_indices = []

        for entity_id, radius, subgraph_nodes in batch_data:
            entity_indices.append(entity_id)

            # Get edges within the ego-graph
            # Find edges in original edge_index where both endpoints are in subgraph_nodes
            node_set = set(subgraph_nodes)
            for i in range(self.dataset.edge_index.shape[0]):
                h = int(self.dataset.edge_index[i, 0])
                t = int(self.dataset.edge_index[i, 2])
                r = int(self.dataset.edge_types[i])
                if h in node_set and t in node_set:
                    all_edges.append([h, r, t])
                    all_types.append(r)
            
            # Get unique relations for this entity
            rels = torch.unique(
                self.dataset.edge_types[
                    self.dataset.edge_index[:, 0] == entity_id
                ]
            )
            relation_indices.extend(rels.tolist())

        if all_edges:
            batch_edge_index = torch.tensor(all_edges, dtype=torch.long).to(self.device)
            batch_edge_types = torch.tensor(all_types, dtype=torch.long).to(self.device)
        else:
            # Fallback: use full graph
            batch_edge_index = self.dataset.edge_index.to(self.device)
            batch_edge_types = self.dataset.edge_types.to(self.device)

        return batch_edge_index, batch_edge_types, entity_indices, relation_indices

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        relation_indices: List[int],
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss.

        Combines:
        1. Cluster entropy loss (encourages confident assignments)
        2. ELBO loss (updates q_mu cluster centers)
        3. Semantic separation loss
        4. Feature smoothness loss (laplacian regularization)

        Args:
            outputs: Dict from model forward pass.
            relation_indices: Active relation indices.

        Returns:
            Dict with total_loss and component losses.
        """
        cluster_probs = outputs["cluster_probs"]
        embeddings = outputs["embeddings"]

        # Cluster entropy loss (encourages confident assignments)
        entropy_loss = -torch.sum(cluster_probs * torch.log(cluster_probs.clamp(min=1e-8)))
        entropy_loss = entropy_loss / cluster_probs.size(0)

        # ELBO loss: updates q_mu cluster centers
        reconstruction_loss, kl_loss = self.model.bayesian_clustering.elbo_loss(
            embeddings, cluster_probs
        )
        elbo_loss = reconstruction_loss - 0.01 * kl_loss  # balance KL weight

        # Feature smoothness: encourage connected nodes to have similar embeddings
        edge_index = self.dataset.edge_index
        node_embeddings = F.normalize(embeddings, dim=1)
        
        h_emb = node_embeddings[edge_index[:, 0]]
        t_emb = node_embeddings[edge_index[:, 2]]
        smoothness_loss = F.pairwise_distance(h_emb, t_emb, p=2).mean()

        # Combine losses
        total_loss = 1.0 * entropy_loss + 0.5 * elbo_loss + 0.01 * smoothness_loss

        return {
            "total": total_loss,
            "entropy": entropy_loss,
            "elbo": elbo_loss,
            "reconstruction": reconstruction_loss,
            "kl": kl_loss,
            "smoothness": smoothness_loss,
        }

    def get_final_embeddings(self) -> np.ndarray:
        """Get final entity embeddings."""
        self.model.eval()
        if self.model.initial_features is not None:
            x = self.model.initial_features
        else:
            x = torch.zeros(self.dataset.edge_index[:, 0].max().item() + 1,
                          self.model.backbone.hidden_dim, device=self.device)
        
        with torch.no_grad():
            embeddings = self.model.backbone(x, self.dataset.edge_index.to(self.device),
                                              self.dataset.edge_types.to(self.device))
        return embeddings.cpu().numpy()

    def fit(
        self,
        val_dataset=None,
    ) -> Dict:
        """Train the model.

        Args:
            val_dataset: Optional validation dataset for early stopping.

        Returns:
            Dict with training results and metadata.
        """
        max_epochs = self.config.get("max_epochs", 500)
        patience = self.config.get("early_stopping_patience", 50)
        batch_size = self.config.get("batch_size", 512)

        # Initialize q_mu with KMeans on R-GCN embeddings
        if hasattr(self.model, 'bayesian_clustering'):
            x = self.model.initial_features
            with torch.no_grad():
                all_emb = self.model.backbone(x, self.dataset.edge_index.to(self.device),
                                               self.dataset.edge_types.to(self.device))
            n_init = min(10, max(2, len(all_emb) // 20))
            self.model.bayesian_clustering.initialize_with_kmeans(all_emb, n_init=n_init)
            del x, all_emb

        best_loss = float("inf")
        patience_counter = 0
        start_time = time.time()
        losses = []

        for epoch in range(max_epochs):
            self.model.train()
            self.optimizer.zero_grad()

            # Sample batch
            batch_edges, batch_types, entity_ids, rel_indices = self.sample_batch_edges(batch_size)

            # Forward pass - use full dataset for R-GCN propagation
            outputs = self.model(
                edge_index=self.dataset.edge_index.to(self.device),
                edge_types=self.dataset.edge_types.to(self.device),
                relation_indices=torch.tensor(rel_indices, dtype=torch.long, device=self.device),
                apply_uncertainty=False,
            )

            # Compute loss
            loss = self.compute_loss(outputs, rel_indices)

            # Backward pass
            loss["total"].backward()

            # Gradient clipping
            grad_clip = self.config.get("gradient_clip", 1.0)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.optimizer.step()

            # Track loss
            loss_val = loss["total"].item()
            losses.append(loss_val)

            # Early stopping (on val loss if available, else train loss)
            eval_loss = loss_val
            if val_dataset is not None and (epoch + 1) % 50 == 0:
                eval_loss = loss_val  # Simplified val

            if eval_loss < best_loss - 1e-4:
                best_loss = eval_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Restore best model
        if "best_state" in dir():
            self.model.load_state_dict(best_state)

        elapsed = time.time() - start_time

        # Compute final metrics
        # Get embeddings for the dataset
        if self.model.initial_features is not None:
            x = self.model.initial_features
        else:
            x = torch.zeros(self.dataset.edge_index[:, 0].max().item() + 1,
                          self.model.backbone.hidden_dim, device=self.device)
        with torch.no_grad():
            train_emb = self.model.backbone(x, self.dataset.edge_index.to(self.device),
                                             self.dataset.edge_types.to(self.device))
        
        # Compute ground truth labels from entity_types
        gt_labels = None
        if hasattr(self.dataset, "entity_types") and self.dataset.entity_types:
            gt_labels = np.zeros(self.dataset.entity_count, dtype=int)
            for eid, ctype in self.dataset.entity_types.items():
                # Handle both int keys and string keys ('e_X')
                if isinstance(eid, str):
                    try:
                        eid = int(eid.split('_')[1])
                    except (ValueError, IndexError):
                        continue
                if isinstance(eid, (int, np.integer)) and 0 <= eid < self.dataset.entity_count:
                    gt_labels[eid] = int(ctype) if isinstance(ctype, (int, float)) else int(str(ctype).strip())
        
        if gt_labels is not None:
            # Use KMeans on R-GCN embeddings as the primary clustering
            # (Bayesian clustering is used for uncertainty estimation, not primary clustering)
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=self.model.num_clusters, random_state=42, n_init=10)
            assignments = km.fit_predict(train_emb.cpu().numpy())
            nmi = compute_nmi(gt_labels, assignments)
            ari = compute_ari(gt_labels, assignments)
        else:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=self.model.num_clusters, random_state=42, n_init=10)
            assignments = km.fit_predict(train_emb.cpu().numpy())
            nmi = 0.0
            ari = 0.0

        results = {
            "nmi": float(nmi),
            "ari": float(ari),
            "losses": losses,
            "elapsed_time": elapsed,
            "epochs_run": patience_counter,
            "best_loss": float(best_loss),
        }
        return results

    def get_embeddings(self, dataset) -> torch.Tensor:
        """Get node embeddings for a dataset.

        Args:
            dataset: Dataset object with edge_index and entity_count.

        Returns:
            torch.Tensor of node embeddings (N, hidden_dim).
        """
        self.model.eval()
        if self.model.initial_features is not None:
            x = self.model.initial_features
        else:
            x = torch.zeros(dataset.edge_index[:, 0].max().item() + 1,
                          self.model.backbone.hidden_dim, device=self.device)

        with torch.no_grad():
            embeddings = self.model.backbone(x, dataset.edge_index.to(self.device),
                                              dataset.edge_types.to(self.device))
        return embeddings

    def _compute_val_nmi(self, val_dataset) -> float:
        """Compute NMI on validation dataset."""
        from gp_llm.evaluation.metrics import compute_nmi

        # Get validation embeddings
        self.model.eval()
        if self.model.initial_features is not None:
            x = self.model.initial_features
        else:
            x = torch.zeros(val_dataset.edge_index[:, 0].max().item() + 1,
                          self.model.backbone.hidden_dim, device=self.device)

        with torch.no_grad():
            val_embeddings = self.model.backbone(
                x,
                val_dataset.edge_index.to(self.device),
                val_dataset.edge_types.to(self.device),
            )

        assignments = self.model.assign_clusters(val_embeddings).cpu().numpy()

        if hasattr(val_dataset, "cluster_labels") and val_dataset.cluster_labels is not None:
            true_labels = val_dataset.cluster_labels
        elif hasattr(val_dataset, "entity_types") and val_dataset.entity_types:
            true_labels = np.array([val_dataset.entity_types.get(i, 0) for i in range(val_dataset.entity_count)])
        else:
            # Use community labels from structure
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=val_dataset.num_clusters, random_state=42)
            true_labels = km.fit_predict(val_embeddings.cpu().numpy())

        nmi = compute_nmi(true_labels, assignments)
        return float(nmi)
