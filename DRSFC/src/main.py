"""
DRSFC: Doubly Regressing Subgroup Fair Clustering

Main entry point for DRSFC experiments.

Usage:
    # Basic usage
    python -m src.main --data_name adult(2) --K 5

    # With hyperparameters
    python -m src.main --data_name adult(4) --K 5 --lambda_fair 10 --lr 0.01

    # Baseline K-Means (unfair)
    python -m src.main --data_name adult(2) --K 5 --unfair
"""

import argparse

parser = argparse.ArgumentParser(description='DRSFC')

# Random seed
parser.add_argument('--seed', default=42, type=int, help='Random seed')

# Data
parser.add_argument('--data_name', default='adult(2)', type=str, help='Dataset name')
parser.add_argument('--data_dir', default=None, type=str, help='Data directory override')

# Clustering
parser.add_argument('--K', default=5, type=int, help='Number of clusters')

# DRSFC hyperparameters
parser.add_argument('--lambda_fair', default=1.0, type=float, help='Fairness regularization weight')
parser.add_argument('--gamma', default=0.01, type=float, help='Minimum subgroup proportion')
parser.add_argument('--max_order', default=2, type=int, help='Maximum interaction order for subgroups')

# Training
parser.add_argument('--max_iter', default=200, type=int, help='Maximum iterations')
parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
parser.add_argument('--n_disc_steps', default=5, type=int, help='Discriminator steps per iteration')

# Options
parser.add_argument('--unfair', action='store_true', help='Use baseline K-Means (unfair)')
parser.add_argument('--use_cuda', action='store_true', help='Use GPU if available')
parser.add_argument('--verbose', action='store_true', help='Verbose output')

args = parser.parse_args()

print('='*60)
print('DRSFC: Doubly Regressing Subgroup Fair Clustering')
print('='*60)
for key, value in vars(args).items():
    print(f'\t [{key}]: {value}')
print('='*60)

if __name__ == "__main__":
    if args.unfair:
        from src.DRSFC import DRSFC
        runner = DRSFC.unfair_runner
    else:
        from src.DRSFC import DRSFC
        runner = DRSFC.runner
    
    runner(args)
    print('='*60)
