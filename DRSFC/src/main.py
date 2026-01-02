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
import sys
from pathlib import Path

# Allow running as a script from inside src/ (e.g., `python main.py ...`).
# This adds the project root so that `import src...` works.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

parser = argparse.ArgumentParser(description='DRSFC')

# Random seed
parser.add_argument('--seed', default=42, type=int, help='Random seed')

# Data
parser.add_argument('--data_name', default='adult(2)', type=str, help='Dataset name')
parser.add_argument('--data_dir', default=None, type=str, help='Data directory override')

# Backward-compatible alias
parser.add_argument('--dataset', dest='data_name', type=str, help='Alias for --data_name')

# Clustering
parser.add_argument('--K', default=5, type=int, help='Number of clusters')

# DRSFC hyperparameters
parser.add_argument('--lambda_fair', default=1.0, type=float, help='Fairness regularization weight')
parser.add_argument('--gamma', default=0.01, type=float, help='Minimum subgroup proportion')
parser.add_argument('--max_order', default=2, type=int, help='Maximum interaction order for subgroups')

# Training
parser.add_argument('--epochs', dest='epochs', default=200, type=int, help='Number of training epochs')
parser.add_argument('--lr', default=0.01, type=float, help='Learning rate (default for all components)')
parser.add_argument('--lr_assign', default=None, type=float, help='Learning rate for assignment network (if None, uses --lr)')
parser.add_argument('--lr_disc', default=None, type=float, help='Learning rate for discriminator (if None, uses --lr)')
parser.add_argument('--lr_v', default=None, type=float, help='Learning rate for v vector (if None, uses --lr). Set to 0 to fix v.')
parser.add_argument('--n_disc_steps', default=5, type=int, help='Discriminator steps per epoch')
parser.add_argument('--warmup_epochs', default=0, type=int, help='Epochs to train discriminator only before adversarial training')
parser.add_argument('--assign_init_epochs', default=0, type=int, help='Epochs to pre-train assignment network on K-means labels')

# Network capacity
parser.add_argument('--hidden_dim', default=1024, type=int, help='Hidden dimension for networks (assignment net / discriminator)')

# Center update and assignment type
parser.add_argument('--center_update', default='mstep', type=str, choices=['sgd', 'mstep', 'fixed'],
                    help='How to update centers: sgd, mstep (closed-form, recommended), or fixed (keep initial)')
parser.add_argument('--assignment_type', default='nn_dist', type=str, choices=['nn', 'distance', 'nn_dist'],
                    help='Assignment type: nn (neural net, x only), distance (softmax of -dist), nn_dist (neural net with distances, recommended)')
parser.add_argument('--tau', default=1.0, type=float, help='Temperature for softmax in assignment (smaller = more peaky)')
parser.add_argument('--softmax_type', default='softmax', type=str, choices=['softmax', 'gumbel'],
                    help='Softmax type: softmax (default) or gumbel (Gumbel-Softmax with straight-through)')

# Center initialization
parser.add_argument('--center_init', default='k-means++', type=str, choices=['k-means++', 'random', 'rand_K'],
                    help='Center init: k-means++ (default), random, or rand_K (random K data points)')
parser.add_argument('--init_centers_path', default=None, type=str,
                    help='Optional path to a numpy .npy file containing initial centers (K,d). If provided, it overrides --center_init.')

# Gradient clipping
parser.add_argument('--grad_clip_disc', default=None, type=float, help='Gradient clipping for discriminator (None = no clipping)')
parser.add_argument('--grad_clip_assign', default=None, type=float, help='Gradient clipping for assignment network (None = no clipping)')
parser.add_argument('--grad_clip_v', default=None, type=float, help='Gradient clipping for v vector (None = no clipping)')

# Cluster size regularization
parser.add_argument('--epsilon_size', default=None, type=str, help='Min cluster proportion (None, "auto", or float like 0.5)')
parser.add_argument('--lambda_size', default=1.0, type=float, help='Weight for cluster size regularization loss')

# Options
parser.add_argument('--unfair', action='store_true', help='Use baseline K-Means (unfair)')
parser.add_argument('--use_cuda', action='store_true', help='(Deprecated) Use GPU if available (GPU is auto by default)')
parser.add_argument('--cpu', action='store_true', help='Force CPU even if CUDA is available')
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
