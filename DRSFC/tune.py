"""
DRSFC Hyperparameter Tuning with Optuna

Bayesian optimization to maximize fairness metrics.

Usage:
    python tune.py --data_name adult(2) --K 5 --objective subgroup
    python tune.py --data_name adult(2) --K_min 2 --K_max 10 --objective subgroup
    python tune.py --data_name adult(2) --K 5 --objective wmp --n_trials 100
"""

import argparse
import json
import os
import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler
import pandas as pd

from src.datasets import load_data
from src.evaluation import compute_all
from src.DRSFC.DRSFC import DRSFC


def parse_args():
    parser = argparse.ArgumentParser(description='DRSFC Hyperparameter Tuning')
    parser.add_argument('--data_name', type=str, default=None)
    parser.add_argument('--dataset', dest='data_name', type=str, help='Alias for --data_name')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--K', type=int, default=None, help='Single K value')
    parser.add_argument('--K_min', type=int, default=2)
    parser.add_argument('--K_max', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--max_order', type=int, default=2)
    parser.add_argument('--lr_min', type=float, default=0.0001)
    parser.add_argument('--lr_max', type=float, default=0.1)
    parser.add_argument('--lambda_min', type=float, default=0.001)
    parser.add_argument('--lambda_max', type=float, default=100)
    parser.add_argument('--epochs_min', dest='epochs_min', type=int, default=100)
    parser.add_argument('--epochs_max', dest='epochs_max', type=int, default=500)
    parser.add_argument('--n_trials', type=int, default=300)
    parser.add_argument('--timeout', type=int, default=None)
    parser.add_argument('--target_value', type=float, default=None)
    parser.add_argument('--target', dest='target_value', type=float, default=None,
                        help='Alias for --target_value')
    parser.add_argument('--objective', type=str, default='subgroup',
                        choices=['marginal', 'subgroup', 'wmp'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_disc_steps', type=int, default=5)

    # Structural options
    # If you pass ONE value -> fixed.
    # If you pass MULTIPLE values -> Optuna will choose among them per trial.
    parser.add_argument(
        '--assignment_type',
        nargs='+',
        default=['nn_dist'],
        choices=['nn', 'distance', 'nn_dist'],
        help=(
            'Assignment type search space. Provide one value to fix it, or multiple values to tune over them. '
            'Examples: --assignment_type distance  |  --assignment_type nn_dist distance'
        )
    )
    parser.add_argument(
        '--center_update',
        nargs='+',
        default=['mstep'],
        choices=['mstep', 'sgd'],
        help=(
            'Center update search space. Provide one value to fix it, or multiple values to tune over them. '
            'Example: --center_update mstep'
        )
    )
    parser.add_argument(
        '--center_init',
        nargs='+',
        default=['k-means++'],
        choices=['k-means++', 'random'],
        help=(
            'Center init search space. Provide one value to fix it, or multiple values to tune over them. '
            'Example: --center_init k-means++'
        )
    )

    parser.add_argument(
        '--fixed_assignment_type',
        type=str,
        default=None,
        choices=['nn', 'distance', 'nn_dist'],
        help='(Deprecated) Use --assignment_type <value> instead.'
    )
    parser.add_argument(
        '--fixed_center_update',
        type=str,
        default=None,
        choices=['mstep', 'sgd'],
        help='(Deprecated) Use --center_update <value> instead.'
    )
    parser.add_argument(
        '--fixed_center_init',
        type=str,
        default=None,
        choices=['k-means++', 'random'],
        help='(Deprecated) Use --center_init <value> instead.'
    )

    args = parser.parse_args()
    if not args.data_name:
        parser.error('one of --data_name or --dataset is required')

    # Apply deprecated fixed_* overrides if provided
    if args.fixed_assignment_type is not None:
        args.assignment_type = [args.fixed_assignment_type]
    if args.fixed_center_update is not None:
        args.center_update = [args.fixed_center_update]
    if args.fixed_center_init is not None:
        args.center_init = [args.fixed_center_init]

    return args


def get_device():
    if not torch.cuda.is_available():
        return 'cpu'
    best_gpu, best_mem = 0, 0
    for i in range(torch.cuda.device_count()):
        try:
            free_mem = torch.cuda.mem_get_info(i)[0]
            if free_mem > best_mem:
                best_gpu, best_mem = i, free_mem
        except:
            continue
    return f'cuda:{best_gpu}'


# Globals for objective
_X, _y, _S, _args, _device, _current_K, _objective_type = [None]*7


def objective(trial):
    """Optuna objective function."""
    global _X, _y, _S, _args, _device, _current_K, _objective_type
    
    lr = trial.suggest_float('lr', _args.lr_min, _args.lr_max, log=True)
    lambda_fair = trial.suggest_float('lambda_fair', _args.lambda_min, _args.lambda_max, log=True)
    # We keep the Optuna parameter name 'max_iter' for continuity in stored studies,
    # but it represents the number of training epochs.
    max_iter = trial.suggest_int('max_iter', _args.epochs_min, _args.epochs_max, step=50)

    # Structural options: if user provided ONE value, fix it; if multiple, tune over that subset.
    if len(_args.assignment_type) == 1:
        assignment_type = _args.assignment_type[0]
    else:
        assignment_type = trial.suggest_categorical('assignment_type', _args.assignment_type)

    if len(_args.center_update) == 1:
        center_update = _args.center_update[0]
    else:
        center_update = trial.suggest_categorical('center_update', _args.center_update)

    if len(_args.center_init) == 1:
        center_init = _args.center_init[0]
    else:
        center_init = trial.suggest_categorical('center_init', _args.center_init)
    
    np.random.seed(_args.seed)
    torch.manual_seed(_args.seed)
    
    try:
        model = DRSFC(
            n_clusters=_current_K,
            lambda_fair=lambda_fair,
            gamma=_args.gamma,
            max_order=_args.max_order,
            lr=lr,
            n_disc_steps=_args.n_disc_steps,
            max_iter=max_iter,
            verbose=False,
            random_state=_args.seed,
            device=_device,
            assignment_type=assignment_type,
            center_update=center_update,
            center_init=center_init,
        )
        model.fit(_X, _S)
        cluster_ids = model.labels_
        soft_assignments = model.assignment_probs_
        centers = model.get_centers()
        
        # Store results for best trial retrieval
        trial.set_user_attr('cluster_ids', cluster_ids.tolist())
        trial.set_user_attr('soft_assignments', soft_assignments.tolist())
        trial.set_user_attr('centers', centers.tolist())
        
        # Compute metrics
        metrics = compute_all(_X, _y, _S, cluster_ids, _current_K, soft_assignments)
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                trial.set_user_attr(key, value)
        
        if _objective_type == 'marginal':
            return metrics['MarginalBalance']
        elif _objective_type == 'wmp':
            return -metrics.get('WMP', float('inf'))
        else:
            return metrics['SubgroupBalance']
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return -float('inf') if _objective_type == 'wmp' else 0.0


def run_tuning_for_K(K, args, X, y, S, device):
    """Run Optuna tuning for a single K value."""
    global _X, _y, _S, _args, _device, _current_K, _objective_type
    
    _X, _y, _S = X, y, S
    _args = args
    _device = device
    _current_K = K
    _objective_type = args.objective
    
    obj_name = {'marginal': 'MarginalBalance', 'subgroup': 'SubgroupBalance', 'wmp': 'WMP'}[args.objective]
    
    print("\n" + "="*70)
    print(f"Tuning K={K} | Objective: {obj_name}")
    print("="*70)
    
    class TargetValueCallback:
        def __init__(self, target_value, objective_type):
            self.target_value = target_value
            self.objective_type = objective_type
        def __call__(self, study, trial):
            if self.target_value is None:
                return
            if self.objective_type == 'wmp':
                if -trial.value <= self.target_value:
                    print(f"\nTarget WMP {self.target_value} reached!")
                    study.stop()
            else:
                if trial.value >= self.target_value:
                    print(f"\nTarget {self.target_value} reached!")
                    study.stop()
    
    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name=f"DRSFC_{args.data_name}_K{K}_{args.objective}"
    )
    
    callback = TargetValueCallback(args.target_value, args.objective)
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        callbacks=[callback],
        show_progress_bar=True
    )
    
    # Best trial
    best = study.best_trial
    
    print("\n" + "-"*50)
    print(f"BEST CONFIG for K={K}")
    print("-"*50)
    print(f"  lr:          {best.params['lr']:.6f}")
    print(f"  lambda_fair: {best.params['lambda_fair']:.4f}")
    print(f"  epochs:      {best.params['max_iter']}")
    if 'assignment_type' in best.params:
        print(f"  assignment:  {best.params['assignment_type']}")
    if 'center_update' in best.params:
        print(f"  center_upd:  {best.params['center_update']}")
    if 'center_init' in best.params:
        print(f"  center_init: {best.params['center_init']}")
    print()
    for m in ['Cost', 'MarginalBalance', 'SubgroupBalance', 'Delta', 'MP1', 'MP2', 'SP']:
        v = best.user_attrs.get(m, 'N/A')
        print(f"  {m}: {v}")
    
    # Retrieve results from best trial
    cluster_ids = np.array(best.user_attrs['cluster_ids'], dtype=np.int32)
    soft_assignments = np.array(best.user_attrs['soft_assignments'], dtype=np.float32)
    centers = np.array(best.user_attrs['centers'], dtype=np.float32)
    
    # Save
    save_dir = f"results/{args.data_name}_tune_{args.objective}/K_{K}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Cluster results (best trial)
    np.save(f"{save_dir}/cluster_ids.npy", cluster_ids)
    np.save(f"{save_dir}/soft_assignments.npy", soft_assignments)
    np.save(f"{save_dir}/centers.npy", centers)
    
    # Data (for reproducibility)
    np.save(f"{save_dir}/X.npy", X)
    np.save(f"{save_dir}/y.npy", y)
    np.save(f"{save_dir}/S.npy", S.values if hasattr(S, 'values') else S)
    
    # best_config.json
    best_config = {
        'data_name': args.data_name,
        'K': K,
        'objective': args.objective,
        'lr': best.params['lr'],
        'lambda_fair': best.params['lambda_fair'],
        'epochs': best.params['max_iter'],
        'assignment_type': best.params.get('assignment_type', None),
        'center_update': best.params.get('center_update', None),
        'center_init': best.params.get('center_init', None),
        'gamma': args.gamma,
        'max_order': args.max_order,
        'seed': args.seed,
        'n_trials': len(study.trials),
        'best_trial_number': best.number,
        'metrics': {
            'Cost': best.user_attrs.get('Cost'),
            'MarginalBalance': best.user_attrs.get('MarginalBalance'),
            'SubgroupBalance': best.user_attrs.get('SubgroupBalance'),
            'Delta': best.user_attrs.get('Delta'),
            'MP1': best.user_attrs.get('MP1'),
            'MP2': best.user_attrs.get('MP2'),
            'SP': best.user_attrs.get('SP'),
        }
    }
    with open(f"{save_dir}/best_config.json", 'w') as f:
        json.dump(best_config, f, indent=2)
    
    # trials CSV
    trials_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            row = {
                'trial': trial.number,
                'lr': trial.params['lr'],
                'lambda_fair': trial.params['lambda_fair'],
                'epochs': trial.params['max_iter'],
                'Cost': trial.user_attrs.get('Cost'),
                'MarginalBalance': trial.user_attrs.get('MarginalBalance'),
                'SubgroupBalance': trial.user_attrs.get('SubgroupBalance'),
                'Delta': trial.user_attrs.get('Delta'),
                'objective_value': trial.value,
            }
            trials_data.append(row)
    pd.DataFrame(trials_data).to_csv(f"{save_dir}/optuna_trials.csv", index=False)
    
    print(f"\nResults saved to: {save_dir}/")
    return best_config


if __name__ == "__main__":
    args = parse_args()
    
    K_list = [args.K] if args.K else list(range(args.K_min, args.K_max + 1))
    obj_name = {'marginal': 'MarginalBalance', 'subgroup': 'SubgroupBalance', 'wmp': 'WMP'}[args.objective]
    
    print("="*70)
    print("DRSFC Hyperparameter Tuning (Optuna)")
    print("="*70)
    print(f"Dataset:   {args.data_name}")
    print(f"Objective: {args.objective} ({obj_name})")
    print(f"K values:  {K_list}")
    print(f"N trials:  {args.n_trials}")
    print("="*70 + "\n")
    
    X, S, y, K_default, d, q = load_data(args.data_name, data_dir=args.data_dir)
    print(f"Data loaded: n={X.shape[0]}, d={d}, q={q}")
    
    device = get_device()
    print(f"Device: {device}\n")
    
    all_results = {}
    for K in K_list:
        all_results[K] = run_tuning_for_K(K, args, X, y, S, device)
    
    # Summary
    print("\n" + "="*80)
    print(f"SUMMARY: Best {obj_name} for each K")
    print("="*80)
    for K in K_list:
        r = all_results[K]
        m = r['metrics']
        print(f"K={K}: {obj_name}={m.get(obj_name.replace('(up)','').replace('(down)',''), 'N/A'):.4f}, "
              f"Cost={m.get('Cost', 'N/A'):.4f}, lr={r['lr']:.6f}, lambda={r['lambda_fair']:.4f}")
    print("="*80)
    
    # Save summary
    summary_dir = f"results/{args.data_name}_tune_{args.objective}"
    os.makedirs(summary_dir, exist_ok=True)
    summary = {
        'data_name': args.data_name,
        'objective': args.objective,
        'K_list': K_list,
        'n_trials': args.n_trials,
        'results': {str(K): all_results[K] for K in K_list}
    }
    with open(f"{summary_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_dir}/summary.json")
