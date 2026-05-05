"""Main training experiment for AdaptKG on a single dataset.

Usage:
    python -m gp_llm.experiments.train --dataset FB15K-237 --num-seeds 10
    python -m gp_llm.experiments.train --dataset WN18RR --num-seeds 10
    python -m gp_llm.experiments.train --dataset YAGO3-10 --num-seeds 10
    python -m gp_llm.experiments.train --dataset PrimeKG --num-seeds 10
"""

import os
import sys
import json
import argparse
import yaml
import numpy as np
import torch

from gp_llm.data.datasets import load_dataset
from gp_llm.data.sampling import DegreeAwareSampler
from gp_llm.models.adaptkg import AdaptKG
from gp_llm.training.trainer import AdaptKGTrainer
from gp_llm.evaluation.metrics import evaluate_clustering, compute_nmi, compute_ari, compute_hits_at_k
from gp_llm.utils.seed import set_seed
from gp_llm.utils.logger import setup_logger


def _flatten_dict(d, prefix=''):
    """Flatten nested dict for config loading."""
    items = {}
    for k, v in d.items():
        key = f'{prefix}_{k}' if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, key))
        else:
            items[key] = v
    return items

def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file or use defaults."""
    defaults = {
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
        'min_radius': 1,
        'max_radius': 4,
        'device': 'cpu',
        'seed': 42,
        'log_dir': '../logs',
        'experiment_name': 'adaptkg',
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        # Flatten nested dict and merge
        flat = _flatten_dict(yaml_config)
        for key, value in flat.items():
            # Only overwrite if the key exists in defaults
            if key in defaults:
                defaults[key] = value
    
    return defaults


def create_config(args, yaml_config: dict = None) -> dict:
    """Create training configuration from args and yaml."""
    if yaml_config is None:
        yaml_config = {}
    
    config = {
        'dataset': args.dataset,
        'embedding_dim': getattr(args, 'embedding_dim', yaml_config.get('embedding_dim', 256)),
        'prompt_length': getattr(args, 'prompt_length', yaml_config.get('prompt_length', 32)),
        'temperature': getattr(args, 'temperature', yaml_config.get('temperature', 0.5)),
        'batch_size': getattr(args, 'batch_size', yaml_config.get('batch_size', 512)),
        'lr': getattr(args, 'lr', yaml_config.get('lr', 1e-3)),
        'max_epochs': getattr(args, 'max_epochs', yaml_config.get('max_epochs', 500)),
        'patience': getattr(args, 'patience', yaml_config.get('patience', 50)),
        'weight_decay': getattr(args, 'weight_decay', yaml_config.get('weight_decay', 1e-5)),
        'dropout': getattr(args, 'dropout', yaml_config.get('dropout', 0.1)),
        'target_volume': getattr(args, 'target_volume', yaml_config.get('target_volume', 100)),
        'num_mc_samples': getattr(args, 'num_mc_samples', yaml_config.get('num_mc_samples', 10)),
        'mmd_weight': getattr(args, 'mmd_weight', yaml_config.get('mmd_weight', 0.1)),
        'mlp_hidden_dims': tuple(yaml_config.get('mlp_hidden_dims', [64, 64])),
        'quantiles': yaml_config.get('quantiles', [0.25, 0.75]),
        'min_radius': getattr(args, 'min_radius', 1),
        'max_radius': getattr(args, 'max_radius', 4),
        'device': getattr(args, 'device', 'cpu'),
        'seed': getattr(args, 'seed', 42),
        'num_seeds': getattr(args, 'num_seeds', 10),
        'data_dir': getattr(args, 'data_dir', yaml_config.get('data_dir', '../data')),
        'log_dir': getattr(args, 'log_dir', yaml_config.get('log_dir', '../logs')),
        'experiment_name': getattr(args, 'experiment_name', yaml_config.get('experiment_name', 'adaptkg')),
    }
    return config


def run_single_seed(
    dataset_name: str,
    seed: int,
    config: dict,
) -> dict:
    """Run AdaptKG for a single random seed.

    Args:
        dataset_name: Name of dataset to train on.
        seed: Random seed.
        config: Training configuration.

    Returns:
        Dictionary with results and metadata.
    """
    # Set seed for reproducibility
    set_seed(seed)

    # Load dataset
    print(f"\n{'='*60}")
    print(f"Seed {seed}: Loading {dataset_name}")
    dataset = load_dataset(dataset_name, data_dir=config['data_dir'])
    print(f"  N={dataset.entity_count}, M={dataset.relation_count}, "
          f"d_bar={dataset.avg_degree:.1f}, K={dataset.num_clusters}")

    # Create model
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

    # Move to device
    device = config['device']
    model = model.to(device)

    # Create trainer
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
            'min_radius': config['min_radius'],
            'max_radius': config['max_radius'],
            'seed': seed,
        },
        device=device,
        seed=seed,
    )

    # Train
    print(f"  Training on {device}...")
    results = trainer.fit()

    # Get final embeddings
    embeddings = trainer.get_final_embeddings()

    # Get cluster assignments
    if model.initial_features is not None:
        x = model.initial_features.to(device)
    else:
        x = torch.zeros(dataset.edge_index[:, 0].max().item() + 1,
                       model.backbone.hidden_dim, device=device)
    with torch.no_grad():
        final_embeddings = model.backbone(x,
                                           dataset.edge_index.to(device),
                                           dataset.edge_types.to(device))
    assignments = model.assign_clusters(final_embeddings.to(device)).cpu().numpy()

    # Get ground truth labels
    if hasattr(dataset, 'cluster_labels') and dataset.cluster_labels is not None:
        true_labels = dataset.cluster_labels
    elif hasattr(dataset, 'entity_types') and dataset.entity_types:
        true_labels = np.array([int(str(dataset.entity_types.get(i, 0)).strip()) for i in range(dataset.entity_count)])
    else:
        true_labels = None

    # Evaluate
    eval_results = evaluate_clustering(
        embeddings,
        true_labels,
        assignments,
        metrics=['nmi', 'ari', 'hits@10'],
    )

    elapsed = results.get('elapsed_seconds', 0)
    print(f"  NMI={eval_results.get('nmi', 0):.4f}, ARI={eval_results.get('ari', 0):.4f}, "
          f"Hits@10={eval_results.get('hits@10', 0):.4f} | "
          f"Time={elapsed:.1f}s")

    return {
        'seed': seed,
        'dataset': dataset_name,
        'nmi': eval_results.get('nmi', 0.0),
        'ari': eval_results.get('ari', 0.0),
        'hits@10': eval_results.get('hits@10', 0.0),
        'elapsed_seconds': elapsed,
        'convergence_epoch': results.get('convergence_epoch'),
        'best_nmi': results.get('best_nmi', -1.0),
        'assignments': assignments.tolist(),
    }


def run_experiment(
    dataset_name: str,
    num_seeds: int = 10,
    config: dict = None,
):
    """Run AdaptKG experiment across multiple seeds.

    Args:
        dataset_name: Dataset name.
        num_seeds: Number of random seeds to run.
        config: Training configuration.
    """
    if config is None:
        config = create_config(
            argparse.Namespace(
                dataset=dataset_name,
                embedding_dim=256,
                prompt_length=32,
                temperature=0.5,
                batch_size=512,
                lr=1e-3,
                max_epochs=500,
                patience=50,
                weight_decay=1e-5,
                dropout=0.1,
                target_volume=100,
                num_mc_samples=10,
                mmd_weight=0.1,
            ),
            None,
        )

    results = []
    for seed in range(num_seeds):
        try:
            result = run_single_seed(dataset_name, seed, config)
            results.append(result)
        except Exception as e:
            print(f"  Seed {seed} failed: {e}")
            results.append({
                'seed': seed,
                'dataset': dataset_name,
                'nmi': -1.0,
                'ari': -1.0,
                'hits@10': -1.0,
                'error': str(e),
            })

    # Summary
    valid_results = [r for r in results if r.get('nmi', -1.0) >= 0]
    if valid_results:
        print(f"\n{'='*60}")
        print(f"Summary for {dataset_name} ({num_seeds} seeds)")
        print(f"  Mean NMI: {np.mean([r['nmi'] for r in valid_results]):.4f} +/- {np.std([r['nmi'] for r in valid_results]):.4f}")
        print(f"  Mean ARI: {np.mean([r['ari'] for r in valid_results]):.4f} +/- {np.std([r['ari'] for r in valid_results]):.4f}")
        print(f"  Mean Hits@10: {np.mean([r['hits@10'] for r in valid_results]):.4f} +/- {np.std([r['hits@10'] for r in valid_results]):.4f}")

    # Save results
    log_dir = config.get('log_dir', '../logs')
    exp_name = config.get('experiment_name', 'adaptkg')
    os.makedirs(log_dir, exist_ok=True)
    output_path = os.path.join(log_dir, f"{exp_name}_results.json")
    with open(output_path, 'w') as f:
        json.dump({
            'dataset': dataset_name,
            'num_seeds': num_seeds,
            'config': config,
            'results': results,
            'mean_nmi': float(np.mean([r['nmi'] for r in valid_results])) if valid_results else -1.0,
            'mean_ari': float(np.mean([r['ari'] for r in valid_results])) if valid_results else -1.0,
        }, f, indent=2)
    print(f"  Results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='AdaptKG Training')
    parser.add_argument('--dataset', type=str, default='FB15K-237',
                        help='Dataset name: FB15K-237, WN18RR, YAGO3-10')
    parser.add_argument('--num-seeds', type=int, default=10,
                        help='Number of random seeds')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')
    parser.add_argument('--embedding-dim', type=int, default=256)
    parser.add_argument('--prompt-length', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max-epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--target-volume', type=int, default=100)
    parser.add_argument('--num-mc-samples', type=int, default=10)
    parser.add_argument('--mmd-weight', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu, cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-dir', type=str, default='../data')
    parser.add_argument('--log-dir', type=str, default='../logs')
    parser.add_argument('--experiment-name', type=str, default='adaptkg')

    args = parser.parse_args()

    # Load YAML config if provided
    yaml_config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f) or {}

    config = create_config(args, yaml_config)
    logger = setup_logger('adaptkg', log_dir=config.get('log_dir'))
    logger.info(f"Starting AdaptKG on {args.dataset} with {args.num_seeds} seeds")
    logger.info(f"Config: {config}")

    results = run_experiment(args.dataset, args.num_seeds, config)
    return results


if __name__ == '__main__':
    main()
