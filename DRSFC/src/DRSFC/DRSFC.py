"""
DRSFC: Doubly Regressing Subgroup Fair Clustering

Main runner and model implementation.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.cluster import KMeans
from tqdm import tqdm
from itertools import combinations

from ..datasets import load_data, get_subgroups
from ..evaluation import evaluation, compute_all, print_metrics


# Core Components

def build_C_matrix(S, gamma=0.01, max_order=2, device="cuda"):
    """Build subgroup-subset indicator matrix C. Values are +1/-1."""
    if isinstance(S, pd.DataFrame):
        S = S.values
    S = np.asarray(S, dtype=np.int32)
    n, q = S.shape
    min_size = int(gamma * n)
    
    subsets = []
    seen = set()
    
    def add_if_valid(mask):
        if mask.sum() >= min_size and (n - mask.sum()) >= min_size:
            key = mask.tobytes()
            if key not in seen:
                seen.add(key)
                subsets.append(mask)
    
    # Order 1: individual attributes
    for j in range(q):
        for val in [0, 1]:
            add_if_valid(S[:, j] == val)
    
    # Order 2+: combinations
    for order in range(2, min(max_order + 1, q + 1)):
        for cols in combinations(range(q), order):
            for pattern in range(2 ** order):
                mask = np.ones(n, dtype=bool)
                for idx, col in enumerate(cols):
                    mask &= (S[:, col] == ((pattern >> idx) & 1))
                add_if_valid(mask)
    
    if len(subsets) == 0:
        return torch.zeros((n, 0), device=device), 0
    
    M = len(subsets)
    C = np.stack([2 * m.astype(np.float32) - 1 for m in subsets], axis=1)
    return torch.tensor(C, dtype=torch.float32, device=device), M


class AssignmentNet(nn.Module):
    """Soft cluster assignment network. Input: x_i, Output: A_ik."""
    
    def __init__(self, n_features, n_clusters, hidden_dim=64, init_tau=1.0, min_tau=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_clusters),
        )
        self._log_tau = nn.Parameter(torch.tensor(np.log(init_tau), dtype=torch.float32))
        self.min_tau = min_tau
    
    @property
    def tau(self):
        return self.min_tau + torch.exp(self._log_tau)
    
    def forward(self, X):
        logits = self.net(X)
        return torch.softmax(logits / self.tau, dim=1)


class Discriminator(nn.Module):
    """Discriminator g(A_ik) for adversarial fairness."""
    
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, A):
        shape = A.shape
        return self.net(A.view(-1, 1)).view(shape)


def clustering_loss(A, X, centers):
    """L_cluster = sum_i sum_k A_ik * ||x_i - mu_k||^2"""
    sq_dist = ((X.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(dim=2)
    return (A * sq_dist).sum(dim=1).mean()


def fairness_loss(A, C, v, g):
    """L_fair = sum_i sum_k (v^T c_i - g(A_ik))^2"""
    vc = C @ v
    gA = g(A)
    return ((vc.unsqueeze(1) - gA) ** 2).mean()


# DRSFC Model

class DRSFC:
    """
    Doubly Regressing Subgroup Fair Clustering.
    
    Objective: min_{A,mu} max_{g,v} L_cluster + lambda * L_fair
    """
    
    def __init__(self, n_clusters=5, lambda_fair=1.0, gamma=0.01, max_order=2,
                 hidden_dim=1024, lr=0.01, n_disc_steps=5, max_iter=200,
                 random_state=42, verbose=True, device="auto"):
        self.n_clusters = n_clusters
        self.lambda_fair = lambda_fair
        self.gamma = gamma
        self.max_order = max_order
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.n_disc_steps = n_disc_steps
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() 
            else device if device != "auto" else "cpu"
        )
        
        self.assignment_net_ = None
        self.centers_ = None
        self.discriminator_ = None
        self.v_ = None
        self.M_ = 0
        self.labels_ = None
        self.assignment_probs_ = None
        self.history_ = None
    
    def fit(self, X, S):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        X = np.asarray(X, dtype=np.float32)
        S = S.values if isinstance(S, pd.DataFrame) else np.asarray(S, dtype=np.float32)
        n, d = X.shape
        K = self.n_clusters
        
        C, self.M_ = build_C_matrix(S, self.gamma, self.max_order, self.device)
        
        if self.verbose:
            print(f"[DRSFC] n={n}, d={d}, K={K}, M={self.M_}, lambda={self.lambda_fair}")
        
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        # Init with K-Means
        km = KMeans(n_clusters=K, n_init=10, random_state=self.random_state)
        km.fit(X)
        self.centers_ = nn.Parameter(
            torch.tensor(km.cluster_centers_, dtype=torch.float32, device=self.device)
        )
        
        # Init networks
        self.assignment_net_ = AssignmentNet(d, K, self.hidden_dim).to(self.device)
        self.discriminator_ = Discriminator(16).to(self.device)
        self.v_ = nn.Parameter(torch.randn(self.M_, device=self.device))
        with torch.no_grad():
            self.v_.data = self.v_.data / (self.v_.data.norm() + 1e-8)
        
        # Optimizers
        opt_main = optim.Adam(
            list(self.assignment_net_.parameters()) + [self.centers_], 
            lr=self.lr
        )
        opt_disc = optim.Adam(self.discriminator_.parameters(), lr=self.lr)
        opt_v = optim.Adam([self.v_], lr=self.lr)
        
        self.history_ = {"cluster_loss": [], "fair_loss": [], "total_loss": []}
        iterator = tqdm(range(self.max_iter), desc="DRSFC") if self.verbose else range(self.max_iter)
        
        for epoch in iterator:
            # Inner loop: maximize L_fair
            if self.lambda_fair > 0 and self.M_ > 0:
                for _ in range(self.n_disc_steps):
                    opt_disc.zero_grad()
                    opt_v.zero_grad()
                    with torch.no_grad():
                        A = self.assignment_net_(X_t)
                    L_fair = fairness_loss(A, C, self.v_, self.discriminator_)
                    L_fair.backward()
                    opt_disc.step()
                    opt_v.step()
                    with torch.no_grad():
                        self.v_.data = self.v_.data / (self.v_.data.norm() + 1e-8)
            
            # Outer loop: minimize L_cluster - lambda * L_fair
            opt_main.zero_grad()
            A = self.assignment_net_(X_t)
            L_cluster = clustering_loss(A, X_t, self.centers_)
            
            if self.lambda_fair > 0 and self.M_ > 0:
                self.discriminator_.eval()
                for p in self.discriminator_.parameters():
                    p.requires_grad_(False)
                L_fair = fairness_loss(A, C, self.v_.detach(), self.discriminator_)
                self.discriminator_.train()
                for p in self.discriminator_.parameters():
                    p.requires_grad_(True)
                total_loss = L_cluster - self.lambda_fair * L_fair
            else:
                L_fair = torch.tensor(0.0)
                total_loss = L_cluster
            
            total_loss.backward()
            opt_main.step()
            
            self.history_["cluster_loss"].append(L_cluster.item())
            self.history_["fair_loss"].append(L_fair.item() if isinstance(L_fair, torch.Tensor) else L_fair)
            self.history_["total_loss"].append(total_loss.item())
        
        with torch.no_grad():
            A = self.assignment_net_(X_t)
            self.labels_ = A.argmax(dim=1).cpu().numpy()
            self.assignment_probs_ = A.cpu().numpy()
        
        if self.verbose:
            print("[DRSFC] Done!")
        
        return self
    
    def predict(self, X):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            A = self.assignment_net_(X_t)
        return A.argmax(dim=1).cpu().numpy()
    
    def get_centers(self):
        return self.centers_.detach().cpu().numpy()


# =============================================================================
# Runners
# =============================================================================

def get_device():
    """Get best available device."""
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


def runner(args):
    """Main DRSFC runner."""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load data
    print("\n[1/4] Loading data...")
    X, S, y, K_default, d, q = load_data(
        args.data_name, 
        l2_normalize=False,
        data_dir=args.data_dir
    )
    K = args.K if args.K > 0 else K_default
    
    # Device
    device = get_device() if args.use_cuda else 'cpu'
    print(f"\n[2/4] Device: {device}")
    
    # Unfair baseline
    print("\n[Unfair] K-Means baseline...")
    km = KMeans(n_clusters=K, random_state=args.seed, n_init=10)
    unfair_labels = km.fit_predict(X)
    unfair_metrics = compute_all(X, y, S, unfair_labels, K)
    print(f"[Unfair] Cost / Balance: {unfair_metrics['Cost']:.4f} / {unfair_metrics['SubgroupBalance']:.4f}")
    
    # DRSFC
    print("\n[3/4] Training DRSFC...")
    model = DRSFC(
        n_clusters=K,
        lambda_fair=args.lambda_fair,
        gamma=args.gamma,
        max_order=args.max_order,
        lr=args.lr,
        n_disc_steps=args.n_disc_steps,
        max_iter=args.max_iter,
        verbose=args.verbose,
        random_state=args.seed,
        device=device,
    )
    model.fit(X, S)
    
    # Evaluate
    print("\n[4/4] Evaluating...")
    metrics = compute_all(X, y, S, model.labels_, K, model.assignment_probs_)
    print_metrics(metrics, title=f"DRSFC on {args.data_name}")
    
    # Save results
    save_dir = Path("results") / f"{args.data_name}_K{K}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Config
    config = {
        'data_name': args.data_name,
        'K': K,
        'lambda_fair': args.lambda_fair,
        'gamma': args.gamma,
        'max_order': args.max_order,
        'lr': args.lr,
        'max_iter': args.max_iter,
        'seed': args.seed,
    }
    with open(save_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Metrics
    with open(save_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Cluster assignments
    np.save(save_dir / "cluster_ids.npy", model.labels_)
    np.save(save_dir / "soft_assignments.npy", model.assignment_probs_)
    np.save(save_dir / "centers.npy", model.get_centers())
    
    print(f"\nResults saved to: {save_dir}/")
    
    # Summary
    print(f"\n[BEST] Cost / Balance: {metrics['Cost']:.4f} / {metrics['SubgroupBalance']:.4f}")


def unfair_runner(args):
    """Baseline K-Means runner (unfair)."""
    np.random.seed(args.seed)
    
    # Load data
    print("\n[1/3] Loading data...")
    X, S, y, K_default, d, q = load_data(
        args.data_name, 
        l2_normalize=False,
        data_dir=args.data_dir
    )
    K = args.K if args.K > 0 else K_default
    
    # K-Means
    print("\n[2/3] Running K-Means...")
    km = KMeans(n_clusters=K, random_state=args.seed, n_init=10)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_
    
    # Evaluate
    print("\n[3/3] Evaluating...")
    metrics = compute_all(X, y, S, labels, K)
    print_metrics(metrics, title=f"K-Means (Unfair) on {args.data_name}")
    
    # Save
    save_dir = Path("results") / f"{args.data_name}_K{K}_unfair"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    np.save(save_dir / "cluster_ids.npy", labels)
    np.save(save_dir / "centers.npy", centers)
    
    print(f"\nResults saved to: {save_dir}/")
    print(f"\n[Result] Cost / Balance: {metrics['Cost']:.4f} / {metrics['SubgroupBalance']:.4f}")
