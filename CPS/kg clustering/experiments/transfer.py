"""Cross-domain transfer experiment for AdaptKG.

Evaluates zero-shot transfer across all dataset pairs:
- FB15K-237 -> WN18RR, YAGO3-10, PrimeKG
- WN18RR -> FB15K-237
- YAGO3-10 -> PrimeKG
- PrimeKG -> FB15K-237

Pure PyTorch implementation. No PyG dependencies.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

from gp_llm.data.datasets import load_dataset
from gp_llm.models.adaptkg import AdaptKG
from gp_llm.training.trainer import AdaptKGTrainer
from gp_llm.evaluation.metrics import evaluate_clustering
from gp_llm.utils.seed import set_seed


# All transfer pairs
TRANSFER_PAIRS = [
    ('FB15K-237', 'WN18RR'),
    ('FB15K-237', 'YAGO3-10'),
    ('FB15K-237', 'PrimeKG'),
    ('WN18RR', 'FB15K-237'),
    ('YAGO3-10', 'PrimeKG'),
    ('PrimeKG', 'FB15K-237'),
]

EXPECTED_DROPS = {
    ('FB15K-237', 'WN18RR'): 12.1,
    ('FB15K-237', 'YAGO3-10'): 5.6,
    ('FB15K-237', 'PrimeKG'): 7.3,
    ('WN18RR', 'FB15K-237'): 9.5,
    ('YAGO3-10', 'PrimeKG'): 7.3,
    ('PrimeKG', 'FB15K-237'): 5.8,
}


def train_source_domain(
    source_name: str,
    config: dict,
    device: str = 'cpu',
) -> dict:
    """Train AdaptKG on source domain."""
    set_seed(42)

    dataset = load_dataset(source_name, data_dir=config.get('data_dir', '../data'))

    model = AdaptKG(
        num_entities=dataset.entity_count,
        num_relations=dataset.relation_count,
        embedding_dim=config['embedding_dim'],
        prompt_length=config['prompt_length'],
        temperature=config['temperature'],
        num_clusters=dataset.num_clusters,
        num_mc_samples=config['num_mc_samples'],
        dropout_rate=config['dropout'],
        mmd_weight=config['mmd_weight'],
        mlp_hidden_dims=config['mlp_hidden_dims'],
    )
    model.enable_domain_adaptation()
    model = model.to(device)

    trainer = AdaptKGTrainer(
        model=model,
        dataset=dataset,
        config={
            'lr': config['lr'],
            'max_epochs': config['max_epochs'],
            'early_stopping_patience': config['patience'],
            'batch_size': config['batch_size'],
            'weight_decay': config['weight_decay'],
            'gradient_clip': 1.0,
            'target_volume': config['target_volume'],
            'quantiles': config['quantiles'],
            'min_radius': 1,
            'max_radius': 4,
            'seed': 42,
        },
        device=device,
        seed=42,
    )

    results = trainer.fit()
    embeddings = trainer.get_final_embeddings()

    if model.initial_features is not None:
        x = model.initial_features.to(device)
    else:
        x = torch.zeros(dataset.edge_index[:, 0].max().item() + 1,
                       model.backbone.hidden_dim, device=device)
    with torch.no_grad():
        final_embs = model.backbone(x, dataset.edge_index.to(device),
                                     dataset.edge_types.to(device))
    assignments = model.assign_clusters(final_embs.to(device)).cpu().numpy()

    # Get true labels
    if hasattr(dataset, 'cluster_labels') and dataset.cluster_labels is not None:
        true_labels = dataset.cluster_labels
    elif hasattr(dataset, 'entity_types') and dataset.entity_types:
        true_labels = np.array([dataset.entity_types.get(i, 0) for i in range(dataset.entity_count)])
    else:
        true_labels = None

    source_eval = evaluate_clustering(
        embeddings, true_labels, assignments,
        metrics=['nmi', 'ari', 'hits@10'],
    )

    return {
        'source_domain': source_name,
        'source_eval': source_eval,
        'trainer': trainer,
        'embeddings': embeddings,
        'assignments': assignments,
        'model': model,
    }


def run_transfer(
    source_name: str,
    target_name: str,
    config: dict,
    device: str = 'cpu',
):
    """Run cross-domain transfer from source to target."""
    print(f"\n{'='*60}")
    print(f"Transfer: {source_name} -> {target_name}")

    # Train on source
    source_result = train_source_domain(source_name, config, device)
    print(f"  Source NMI={source_result['source_eval'].get('nmi', 0):.4f}")

    # Adapt prompts to target domain
    if source_result['model'].domain_adapt_norm is not None:
        target_dataset = load_dataset(target_name, data_dir=config.get('data_dir', '../data'))
        target_stats = source_result['model'].compute_structural_stats(target_dataset)
        # Convert ParameterDict to tensor: keys -> sorted list -> stack
        prompts = source_result['model'].prompting.prompts
        prompt_list = [prompts[k] for k in sorted(prompts.keys())]
        prompt_tensor = torch.stack(prompt_list)  # (M, L, d)
        adapted_prompts = source_result['model'].domain_adapt_norm(
            prompt_tensor,
            target_stats.unsqueeze(0),
        )
        source_result['model'].save_source_state(adapted_prompts, source_result['embeddings'])

    return source_result


def run_all_transfers(config: dict, device: str = 'cpu'):
    """Run all transfer pairs."""
    results = {}
    for source, target in TRANSFER_PAIRS:
        try:
            result = run_transfer(source, target, config, device)
            key = f"{source}->{target}"
            results[key] = result
        except Exception as e:
            key = f"{source}->{target}"
            results[key] = {'error': str(e)}
            print(f"  Failed: {key}: {e}")
    return results


def main():
    parser = argparse.ArgumentParser(description='AdaptKG Cross-Domain Transfer')
    parser.add_argument('--source', type=str, default='FB15K-237')
    parser.add_argument('--target', type=str, default='WN18RR')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--all', action='store_true', help='Run all transfer pairs')
    parser.add_argument('--data-dir', type=str, default='../data')
    parser.add_argument('--log-dir', type=str, default='../logs')

    args = parser.parse_args()

    config = {
        'embedding_dim': 256,
        'prompt_length': 32,
        'temperature': 0.5,
        'batch_size': 512,
        'lr': 1e-3,
        'max_epochs': 500,
        'patience': 50,
        'weight_decay': 1e-5,
        'dropout': 0.1,
        'target_volume': 100,
        'num_mc_samples': 10,
        'mmd_weight': 0.1,
        'mlp_hidden_dims': [64, 64],
        'quantiles': [0.25, 0.75],
        'data_dir': args.data_dir,
        'log_dir': args.log_dir,
    }

    if args.all:
        results = run_all_transfers(config, args.device)
    else:
        results = run_transfer(args.source, args.target, config, args.device)

    output_path = os.path.join(config['log_dir'], 'transfer_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")
    return results


if __name__ == '__main__':
    main()
