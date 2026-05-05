"""AdaptKG: Domain-Adaptive Prompting for Unsupervised Knowledge Graph Clustering.

Pure PyTorch implementation (no PyG dependencies).

Reference:
    AdaptKG: Domain-Adaptive Prompting for Unsupervised Knowledge Graph Clustering
    NeurIPS 2026 (under review)

Usage:
    python -m gp_llm.experiments.train --dataset FB15K-237 --num-seeds 10
    python -m gp_llm.experiments.ablation --dataset FB15K-237 --num-seeds 3
    python -m gp_llm.experiments.transfer --source FB15K-237 --target WN18RR
"""

__version__ = "0.1.0"
