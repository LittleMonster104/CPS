"""Pure PyTorch R-GCN backbone with basis decomposition.

No PyG dependencies. Implements the R-GCN layer from:
Schlichtkrull et al., Modeling Relational Data with GCN (2018)

Architecture:
- Basis decomposition with num_bases=5
- 3 layers, hidden_dim=256, ReLU activation
- LayerNorm normalization
- W_r = sum_k alpha_{r,k} b_k
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RGCNLayer(nn.Module):
    """Single R-GCN layer with basis decomposition.

    Message passing:
    h_e^(l+1) = sigma(
        sum_r sum_{(e,r,t) in E_r} (1/c_{e,r}) * W_r * h_e^(l)
        + W_0 * h_e^(l)
    )

    Where W_r = sum_k alpha_{r,k} * b_k (basis decomposition)
    and c_{e,r} is the number of edges with relation r incident to e.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        num_bases: int = 5,
        bias: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        # Basis matrices: (num_bases, in_channels, out_channels)
        self.basis = nn.Parameter(torch.randn(num_bases, in_channels, out_channels) * 0.01)

        # Secondary matrices for offset: (num_relations, in_channels, out_channels)
        self.s_base = nn.Parameter(torch.zeros(num_relations, in_channels, out_channels))

        # Relation attention weights: (num_relations, num_bases)
        self.att = nn.Parameter(torch.ones(num_relations, num_bases) / num_bases)

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # Layer norm
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # (N, in_channels)
        edge_index: torch.Tensor,  # (E, 3): [head, relation, tail]
        edge_types: torch.Tensor,  # (E,)
    ) -> torch.Tensor:
        """Forward pass through R-GCN layer.

        Args:
            x: Node features (N, in_channels).
            edge_index: Edge list (E, 3) with [head, relation, tail].
            edge_types: Relation type for each edge (E,).

        Returns:
            Output node embeddings (N, out_channels).
        """
        # Compute relation-specific weight matrices via basis decomposition
        # W_r = sum_k att[r,k] * b_k + s_base[r]
        # shape: (num_relations, in_channels, out_channels)
        W_all = torch.einsum('kb,bio->kio', self.att, self.basis) + self.s_base  # (M, d_in, d_out)

        # Compute normalization constants c_{e,r} (degree per relation)
        # c_{hr} = number of edges with relation r incident to head node h
        c_hr = self._compute_normalization(edge_index, edge_types, x.size(0))  # (N, M)

        # Build adjacency structures for message passing
        head_nodes = edge_index[:, 0]  # (E,)
        tail_nodes = edge_index[:, 2]  # (E,)
        rel_types = edge_types  # (E,)

        # Message aggregation per relation
        out_nodes = torch.zeros(x.size(0), self.out_channels, device=x.device, dtype=x.dtype)
        count = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

        # Process each relation separately for efficiency
        unique_rels = edge_types.unique()
        
        for r in unique_rels:
            r = int(r)
            mask = rel_types == r
            if not mask.any():
                continue

            r_head = head_nodes[mask]
            r_tail = tail_nodes[mask]

            # Relation-specific weight matrix: (in_channels, out_channels)
            W_r = W_all[r]  # (d_in, d_out)

            # Gather neighbor features: (num_rel_edges, in_channels)
            neighbor_features = x[r_tail]  # (E_r, d_in)

            # Apply relation weight: (E_r, d_out)
            messages = neighbor_features @ W_r  # (E_r, d_out)

            # Normalize by c_{hr}: each message scaled by 1/c_{hr}
            # c_hr[r_head] gives normalization for each edge
            norm_vals = c_hr[r_head, r].clamp(min=1.0).unsqueeze(-1)  # (E_r, 1)
            messages = messages / norm_vals

            # Add to output with scatter
            out_nodes.index_add_(0, r_head, messages)
            count.index_add_(0, r_head, torch.ones_like(r_head, dtype=x.dtype))

        # Self-loop with W_0 (use basis matrix 0 as W_0)
        W_0 = W_all[0] if 0 in unique_rels else W_all[0]
        self_messages = x @ W_0  # (N, d_out)
        out_nodes = out_nodes + self_messages

        # Add bias and apply activation
        if self.bias is not None:
            out_nodes = out_nodes + self.bias

        out_nodes = self.norm(out_nodes)
        out_nodes = F.relu(out_nodes)
        out_nodes = self.dropout(out_nodes)

        return out_nodes

    def _compute_normalization(
        self,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Compute normalization constants c_{hr}.

        Args:
            edge_index: (E, 3) edge list.
            edge_types: (E,) relation types.
            num_nodes: Number of nodes N.

        Returns:
            c_hr: (N, num_relations) matrix of edge counts per relation.
        """
        # Use scatter_add to count edges per (head, relation) pair
        head_nodes = edge_index[:, 0]
        rel_types = edge_types

        # Create a unique key for each (node, relation) pair
        max_rel = rel_types.max().item() + 1
        keys = head_nodes * max_rel + rel_types
        counts = torch.ones_like(keys, dtype=torch.float32)

        c = torch.zeros(num_nodes * max_rel, device=keys.device, dtype=torch.float32)
        c = c.index_add(0, keys, counts)
        c = c.view(num_nodes, max_rel)

        return c


class RGCNBackbone(nn.Module):
    """3-layer Relational Graph Convolutional Network with basis decomposition.

    Matches AdaptKG implementation details:
    - 3 layers, hidden_dim=256, num_bases=5, ReLU activation
    - Basis decomposition for relation-specific weights
    - LayerNorm normalization
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_relations: int = 237,
        num_layers: int = 3,
        num_bases: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim
            layers.append(RGCNLayer(
                in_channels=in_dim,
                out_channels=out_dim,
                num_relations=num_relations,
                num_bases=num_bases,
                dropout=dropout,
            ))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(
        self,
        x: torch.Tensor,  # (N, input_dim) - initial node features
        edge_index: torch.Tensor,  # (E, 3) edge list [head, relation, tail]
        edge_types: torch.Tensor,  # (E,) relation type for each edge
    ) -> torch.Tensor:
        """Forward pass through R-GCN.

        Args:
            x: Node features (N, input_dim).
            edge_index: Edge list (E, 3) with [head, relation, tail].
            edge_types: Relation type for each edge (E,).

        Returns:
            Node embeddings (N, hidden_dim).
        """
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, edge_types)
            if i < self.num_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def get_initial_features(self, num_entities: int) -> torch.nn.Parameter:
        """Create zero-initialized learnable node features.

        Args:
            num_entities: Number of entities N.

        Returns:
            Learnable node features (N, hidden_dim).
        """
        return nn.Parameter(torch.randn(num_entities, self.hidden_dim) * 0.01)
