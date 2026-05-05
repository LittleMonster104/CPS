"""Ablation study for AdaptKG component contributions.

Evaluates the contribution of each component:
1. Full AdaptKG (all components)
2. -Degree-Aware Sampling
3. -Relation-Specific Prompting
4. -Bayesian Uncertainty
5. -Domain Adaptation

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


# Ablation configurations
ABLATION_CONFIGS = {
    'full_adaptkg': {
        'disable_sampling': False,
        'disable_prompting': False,
        'disable_bayesian': False,
        'disable_domain_adapt': False,
        'description': 'Full AdaptKG',
    },
    'no_degree_sampling': {
        'disable_sampling': True,
        'disable_prompting': False,
        'disable_bayesian': False,
        'disable_domain_adapt': False,
        'description': 'w/o Degree-Aware Sampling',
    },
    'no_relation_prompting': {
        'disable_sampling': False,
        'disable_prompting': True,
        'disable_bayesian': False,
        'disable_domain_adapt': False,
        'description': 'w/o Relation-Specific Prompting',
    },
    'no_bayesian': {
        'disable_sampling': False,
        'disable_prompting': False,
        'disable_bayesian': True,
        'disable_domain_adapt': False,
        'description': 'w/o Bayesian Uncertainty',
    },
    'no_domain_adapt': {
        'disable_sampling': False,
        'disable_prompting': False,
        'disable_bayesian': False,
        'disable_domain_adapt': True,
        'description': 'w/o Domain Adaptation',
    },
}


def create_config(base_config: dict, ablation_cfg: dict) -> dict:
    """Create config for a specific ablation configuration."""
    config = base_config.copy()
    config['disable_sampling'] = ablation_cfg['disable_sampling']
    config['disable_prompting'] = ablation_cfg['disable_prompting']
    config['disable_bayesian'] = ablation_cfg['disable_bayesian']
    config['disable_domain_adapt'] = ablation_cfg['disable_domain_adapt']
    return config


def run_ablation_variant(
    dataset_name: str,
    ablation_key: str,
    ablation_cfg: dict,
    num_seeds: int = 3,
    base_config: dict = None,
    device: str = 'cpu',
) -> dict:
    """Run a single ablation variant across multiple seeds."""
    if base_config is None:
        base_config = {
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
            'data_dir': '../data',
            'log_dir': '../logs',
        }

    config = create_config(base_config, ablation_cfg)
    results = []

    for seed in range(num_seeds):
        try:
            set_seed(seed)
            dataset = load_dataset(dataset_name, data_dir=config['data_dir'])

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
                    'seed': seed,
                },
                device=device,
                seed=seed,
            )

            results_trained = trainer.fit()
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

            if hasattr(dataset, 'cluster_labels') and dataset.cluster_labels is not None:
                true_labels = dataset.cluster_labels
            elif hasattr(dataset, 'entity_types') and dataset.entity_types:
                true_labels = np.array([dataset.entity_types.get(i, 0) for i in range(dataset.entity_count)])
            else:
                true_labels = None

            eval_results = evaluate_clustering(
                embeddings, true_labels, assignments,
                metrics=['nmi', 'ari'],
            )

            results.append({
                'seed': seed,
                'nmi': eval_results.get('nmi', 0.0),
                'ari': eval_results.get('ari', 0.0),
                'loss': results_trained.get('final_loss', 0.0),
            })

            print(f"  Seed {seed}: NMI={eval_results.get('nmi', 0):.4f}, ARI={eval_results.get('ari', 0):.4f}")

        except Exception as e:
            print(f"  Seed {seed} failed: {e}")
            results.append({'seed': seed, 'nmi': -1.0, 'ari': -1.0, 'error': str(e)})

    # Summary
    valid = [r for r in results if r.get('nmi', -1.0) >= 0]
    summary = {
        'ablation': ablation_key,
        'description': ablation_cfg['description'],
        'num_seeds': num_seeds,
        'mean_nmi': float(np.mean([r['nmi'] for r in valid])) if valid else -1.0,
        'mean_ari': float(np.mean([r['ari'] for r in valid])) if valid else -1.0,
        'std_nmi': float(np.std([r['nmi'] for r in valid])) if valid else 0.0,
        'results': results,
    }

    return summary


def run_all_ablations(
    dataset_name: str,
    num_seeds: int = 3,
    base_config: dict = None,
    device: str = 'cpu',
):
    """Run all ablation variants."""
    all_results = {}
    for ablation_key, ablation_cfg in ABLATION_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Ablation: {ablation_key} - {ablation_cfg['description']}")
        summary = run_ablation_variant(
            dataset_name, ablation_key, ablation_cfg,
            num_seeds=num_seeds, base_config=base_config, device=device,
        )
        all_results[ablation_key] = summary
        print(f"  Mean NMI={summary['mean_nmi']:.4f}, Mean ARI={summary['mean_ari']:.4f}")

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"Ablation Summary for {dataset_name}")
    print(f"{'Ablation':<35} {'NMI':>8} {'ARI':>8}")
    print(f"{'-'*51}")
    for key, result in all_results.items():
        print(f"{result['description']:<35} {result['mean_nmi']:>8.4f} {result['mean_ari']:>8.4f}")

    # Save results
    log_dir = base_config.get('log_dir', '../logs') if base_config else '../logs'
    os.makedirs(log_dir, exist_ok=True)
    output_path = os.path.join(log_dir, 'ablation_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='AdaptKG Ablation Study')
    parser.add_argument('--dataset', type=str, default='FB15K-237')
    parser.add_argument('--num-seeds', type=int, default=3)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--data-dir', type=str, default='../data')
    parser.add_argument('--log-dir', type=str, default='../logs')

    args = parser.parse_args()

    base_config = {
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

    results = run_all_ablations(args.dataset, args.num_seeds, base_config, args.device)
    return results


if __name__ == '__main__':
    main()
