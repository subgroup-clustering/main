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

try:
    from ..datasets import load_data, get_subgroups
    from ..evaluation import (
        evaluation, compute_all, print_metrics,
        compute_cost, compute_subgroup_balance, compute_delta,
        compute_gap, compute_SP, compute_MP
    )
except ImportError:
    from datasets import load_data, get_subgroups
    from evaluation import (
        evaluation, compute_all, print_metrics,
        compute_cost, compute_subgroup_balance, compute_delta,
        compute_gap, compute_SP, compute_MP
    )


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


def gumbel_softmax(logits, tau=1.0, hard=True):
    """
    Gumbel-Softmax: Differentiable approximation to discrete sampling.
    
    Args:
        logits: (n, K) unnormalized log-probabilities
        tau: Temperature. Lower = more discrete-like.
        hard: If True, use straight-through estimator (hard forward, soft backward)
    
    Returns:
        (n, K) soft or hard assignments (differentiable)
    """
    # Sample Gumbel noise
    U = torch.rand_like(logits)
    g = -torch.log(-torch.log(U + 1e-10) + 1e-10)
    
    # Add Gumbel noise and apply softmax
    y_soft = torch.softmax((logits + g) / tau, dim=-1)
    
    if hard:
        # Straight-through: forward uses hard, backward uses soft gradient
        y_hard = torch.zeros_like(logits)
        y_hard.scatter_(-1, y_soft.argmax(dim=-1, keepdim=True), 1.0)
        return y_hard - y_soft.detach() + y_soft
    
    return y_soft


class AssignmentNet(nn.Module):
    """
    Soft cluster assignment network.
    
    Modes:
        - "nn": A = f(x) - only sees x
        - "nn_dist": A = f(x, d(x,μ)) - sees x and distances to centers
    
    Args:
        tau: Temperature for softmax. Fixed value (not learnable).
             Smaller tau → more peaky (closer to hard assignment).
             Larger tau → softer assignments.
        softmax_type: Type of softmax to use.
             - "softmax": Standard softmax (default, current behavior)
             - "gumbel": Gumbel-Softmax with straight-through estimator
    """
    
    def __init__(self, n_features, n_clusters, hidden_dim=64, tau=1.0, mode="nn", softmax_type="softmax"):
        super().__init__()
        self.mode = mode
        self.n_clusters = n_clusters
        self.tau = tau  # Fixed, not learnable
        self.softmax_type = softmax_type
        
        if mode == "nn":
            input_dim = n_features
        else:  # "nn_dist"
            input_dim = n_features + n_clusters
        
        # 2-layer structure for stable adversarial training
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_clusters),
        )
    
    def forward(self, X, centers=None):
        """
        Args:
            X: (n, d) data points
            centers: (K, d) cluster centers (required if mode="nn_dist")
        Returns:
            A: (n, K) soft assignments
        """
        if self.mode == "nn":
            features = X
        else:  # "nn_dist"
            distances = torch.sqrt(((X.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(dim=2) + 1e-8)
            features = torch.cat([X, distances], dim=1)
        
        logits = self.net(features)
        
        if self.softmax_type == "gumbel":
            return gumbel_softmax(logits, tau=self.tau, hard=True)
        else:  # "softmax" (default)
            return torch.softmax(logits / self.tau, dim=1)


class Discriminator(nn.Module):
    """Discriminator g(A_i) for adversarial fairness.
    
    Takes the full assignment vector A_i (K-dimensional) and outputs a scalar.
    2-layer structure for stable adversarial training.
    """
    
    def __init__(self, n_clusters, hidden_dim=64):
        super().__init__()
        # 2-layer structure for stable adversarial training
        self.net = nn.Sequential(
            nn.Linear(n_clusters, hidden_dim),  # Input: K-dimensional assignment vector
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Output: scalar
        )
    
    def forward(self, A):
        """
        Args:
            A: (n, K) soft assignment matrix
        Returns:
            (n,) scalar output for each data point
        """
        return self.net(A).squeeze(-1)  # (n, K) -> (n,)


def clustering_loss(A, X, centers):
    """L_cluster = sum_i sum_k A_ik * ||x_i - mu_k||^2"""
    sq_dist = ((X.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(dim=2)
    return (A * sq_dist).sum(dim=1).mean()


def fairness_loss(A, C, v, g):
    """L_fair = sum_i (v^T c_i - g(A_i))^2
    
    Args:
        A: (n, K) soft assignment matrix
        C: (n, M) subgroup indicator matrix
        v: (M,) learnable projection vector
        g: Discriminator network that takes A_i (K-dim) and outputs scalar
    
    Returns:
        scalar loss
    """
    vc = C @ v  # (n,) - projected subgroup membership for each point
    gA = g(A)   # (n,) - discriminator output for each point's assignment vector
    return ((vc - gA) ** 2).mean()


def cluster_size_loss(A, epsilon=None):
    """
    Compute cluster size regularization loss.
    
    Method 2 (Hinge): L_size = sum_k max(0, epsilon - p_k)
    
    This loss penalizes clusters that have less than epsilon proportion of soft assignments.
    By minimizing this loss, the assignment network learns to avoid cluster collapse.
    
    Args:
        A: (n, K) soft assignment matrix (each row sums to 1)
        epsilon: Minimum proportion per cluster, or alpha value for adaptive epsilon.
            - If None: epsilon = 1/K (uniform proportion)
            - If 0 < epsilon <= 1: epsilon = epsilon / K (interpreted as alpha, proportion of uniform)
            - If epsilon > 1: use as-is (legacy behavior, not recommended)
    
    Returns:
        scalar loss (0 if all clusters have >= epsilon proportion)
    """
    n, K = A.shape
    
    if epsilon is None:
        epsilon = 1.0 / K  # Default: each cluster should have at least uniform proportion
    elif epsilon <= 1.0:
        # Interpret as alpha: epsilon = alpha / K
        # alpha=0.9 means each cluster should have at least 90% of uniform proportion
        epsilon = epsilon / K
    # else: use epsilon as-is (legacy behavior for values > 1)
    
    # Current proportions: p_k = 1/n * sum_i A_ik
    p_k = A.mean(dim=0)  # (K,)
    
    # Hinge loss: sum_k max(0, epsilon - p_k)
    loss = torch.sum(torch.relu(epsilon - p_k))
    
    return loss


# DRSFC Model

class DRSFC:
    """
    Doubly Regressing Subgroup Fair Clustering.
    
    Objective: min_{A,mu} max_{g,v} L_cluster - lambda * L_fair
    """
    
    def __init__(self, n_clusters=5, lambda_fair=1.0, gamma=0.01, max_order=2,
                 hidden_dim=1024, lr=0.001, lr_assign=None, lr_disc=None, lr_v=None,
                 n_disc_steps=5, max_iter=200, warmup_epochs=0,
                 assign_init_epochs=0,
                 random_state=42, verbose=True, device="auto",
                 center_update="mstep", assignment_type="nn",
                 center_init="k-means++", init_centers=None, tau=1.0,
                 grad_clip_disc=None, grad_clip_assign=None, grad_clip_v=None,
                 epsilon_size=None, lambda_size=1.0, softmax_type="softmax"):
        """
        Args:
            n_clusters: Number of clusters K.
            lambda_fair: Fairness regularization weight.
            gamma: Minimum subgroup proportion for C matrix.
            max_order: Maximum interaction order for subgroups.
            hidden_dim: Hidden dimension for networks.
            lr: Default learning rate (used if specific lr_* is not set).
            lr_assign: Learning rate for assignment network (and centers if SGD).
                If None, uses lr. Typically set lower than lr_disc for stable adversarial training.
            lr_disc: Learning rate for discriminator.
                If None, uses lr. Can be set higher to let discriminator learn faster.
            lr_v: Learning rate for v (subgroup projection vector).
                If None, uses lr.
            n_disc_steps: Number of discriminator/v steps per epoch (inner loop).
            max_iter: Number of training epochs.
            warmup_epochs: Number of epochs to train discriminator only (before adversarial training).
                During warmup, assignment network is frozen and only discriminator/v are trained.
                This helps discriminator become strong enough before the adversarial game begins.
                Default: 0 (no warmup). Recommended: 10-50 for unstable training.
            assign_init_epochs: Number of epochs to pre-train assignment network on K-means labels.
                This initializes the NN to output assignments matching K-means clustering.
                Only used when assignment_type is "nn" or "nn_dist".
                Default: 0 (no pre-training). Recommended: 50-100 for stable initialization.
            random_state: Random seed.
            verbose: Whether to print progress.
            device: Device to use ("auto", "cpu", or "cuda:X").
            center_update: How to update centers.
                - "sgd": Update via gradient descent (original, unstable).
                - "mstep": Closed-form M-step (recommended).
                - "fixed": Keep centers fixed at initial KMeans solution (for ablation study).
            assignment_type: How to compute soft assignments A.
                - "nn": A = f_θ(x) - neural net that only sees x.
                - "distance": A = softmax(-||x-μ||²/τ) - no learnable function.
                - "nn_dist": A = f_θ(x, d(x,μ)) - neural net sees x AND distances.
            center_init: KMeans init strategy for initializing centers when init_centers is not given.
                - "k-means++": k-means++ seeding (sklearn default)
                - "random": random points seeding
                - "rand_K": randomly select K data points from the dataset as initial centers (no KMeans)
            init_centers: Optional initial centers. Either a numpy array (K,d) or a torch tensor (K,d).
                If provided, it overrides center_init and no KMeans initialization is run.
            tau: Temperature for softmax in assignment. Fixed value (not learnable).
                Smaller tau (e.g., 0.01) → more peaky, closer to hard assignment.
                Larger tau (e.g., 1.0, 10.0) → softer assignments.
            grad_clip_disc: Maximum gradient norm for discriminator gradient clipping.
                If None (default), no clipping is applied.
                Recommended values: 0.5 ~ 2.0 for stabilizing adversarial training.
                This prevents discriminator from becoming too strong too fast.
            grad_clip_assign: Maximum gradient norm for assignment network gradient clipping.
                If None (default), no clipping is applied.
                Recommended values: 5.0 ~ 20.0 for stabilizing training when gradients explode.
            grad_clip_v: Maximum gradient norm for v (subgroup projection) gradient clipping.
                If None (default), no clipping is applied.
                Recommended values: 2.0 ~ 3.0 based on empirical analysis.
            epsilon_size: Minimum proportion per cluster for cluster size regularization.
                If not None, adds L_size = sum_k max(0, epsilon - p_k) to the loss.
                If None (default), no size regularization is applied.
                If 'auto', uses 1/K (uniform proportion) as the minimum.
                Example: epsilon_size=0.05 means each cluster should have at least 5% of assignments.
            lambda_size: Weight for cluster size regularization loss.
                Default: 1.0. Higher values enforce stricter cluster size balance.
                Only used when epsilon_size is not None.
            softmax_type: Type of softmax to use for assignment.
                - "softmax": Standard softmax (default). Soft assignments, may have train/eval gap.
                - "gumbel": Gumbel-Softmax with straight-through estimator.
                    Hard assignments in forward pass, soft gradients in backward.
                    Better alignment between training and hard assignment evaluation.
        """
        self.n_clusters = n_clusters
        self.lambda_fair = lambda_fair
        self.gamma = gamma
        self.max_order = max_order
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.lr_assign = lr_assign if lr_assign is not None else lr
        self.lr_disc = lr_disc if lr_disc is not None else lr
        self.lr_v = lr_v if lr_v is not None else lr
        self.n_disc_steps = n_disc_steps
        self.max_iter = max_iter
        self.warmup_epochs = warmup_epochs
        self.assign_init_epochs = assign_init_epochs
        self.random_state = random_state
        self.verbose = verbose
        self.center_update = center_update
        self.assignment_type = assignment_type
        self.center_init = center_init
        self.init_centers = init_centers
        self.tau = tau
        self.grad_clip_disc = grad_clip_disc
        self.grad_clip_assign = grad_clip_assign
        self.grad_clip_v = grad_clip_v
        self.epsilon_size = epsilon_size
        self.lambda_size = lambda_size
        self.softmax_type = softmax_type
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
    
    def fit(self, X, S, track_metrics=True):
        """
        Fit DRSFC model.
        
        Args:
            X: Data matrix (n, d)
            S: Sensitive attributes (n, q)
            track_metrics: If True, compute Cost and Balance per epoch (slower but useful for analysis)
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        X_np = np.asarray(X, dtype=np.float32)
        S_np = S.values if isinstance(S, pd.DataFrame) else np.asarray(S, dtype=np.float32)
        n, d = X_np.shape
        K = self.n_clusters
        
        C, self.M_ = build_C_matrix(S_np, self.gamma, self.max_order, self.device)
        
        if self.verbose:
            print(f"[DRSFC] n={n}, d={d}, K={K}, M={self.M_}, lambda_fair={self.lambda_fair}, lambda_size={self.lambda_size}, epsilon_size={self.epsilon_size}, softmax_type={self.softmax_type}")
        
        X_t = torch.tensor(X_np, dtype=torch.float32, device=self.device)
        
        # Initialize centers μ
        if self.init_centers is not None:
            init = self.init_centers
            if isinstance(init, torch.Tensor):
                init_centers = init.detach().to(self.device).to(torch.float32)
            else:
                init_centers = torch.tensor(np.asarray(init, dtype=np.float32), device=self.device)
            if init_centers.ndim != 2 or init_centers.shape[0] != K or init_centers.shape[1] != d:
                raise ValueError(
                    f"init_centers must have shape (K,d)=({K},{d}), got {tuple(init_centers.shape)}"
                )
            self.centers_ = nn.Parameter(init_centers)
            km_labels = None  # No K-means labels when using init_centers
        elif self.center_init == "rand_K":
            # Randomly select K data points from the dataset as initial centers
            rng = np.random.default_rng(self.random_state)
            rand_indices = rng.choice(n, size=K, replace=False)
            init_centers = X_np[rand_indices]
            self.centers_ = nn.Parameter(
                torch.tensor(init_centers, dtype=torch.float32, device=self.device)
            )
            km_labels = None  # No K-means labels when using rand_K
            if self.verbose:
                print(f"[DRSFC] Initialized centers with rand_K: selected {K} random data points")
        else:
            km = KMeans(
                n_clusters=K,
                init=self.center_init,
                n_init=10,
                random_state=self.random_state,
            )
            km.fit(X)
            self.centers_ = nn.Parameter(
                torch.tensor(km.cluster_centers_, dtype=torch.float32, device=self.device)
            )
            km_labels = km.labels_  # Save K-means labels for assignment initialization
        
        # Init networks based on assignment_type
        if self.assignment_type in ("nn", "nn_dist"):
            self.assignment_net_ = AssignmentNet(
                d, K, self.hidden_dim, tau=self.tau, mode=self.assignment_type,
                softmax_type=self.softmax_type
            ).to(self.device)
            
            # ========== ASSIGNMENT NETWORK INITIALIZATION ==========
            # Pre-train assignment network to match K-means labels
            if self.assign_init_epochs > 0 and km_labels is not None:
                if self.verbose:
                    print(f"[DRSFC] Pre-training assignment network on K-means labels ({self.assign_init_epochs} epochs)")
                
                # Convert K-means labels to one-hot targets
                km_labels_t = torch.tensor(km_labels, dtype=torch.long, device=self.device)
                
                # Use CrossEntropyLoss for classification
                ce_loss = nn.CrossEntropyLoss()
                opt_init = optim.Adam(self.assignment_net_.parameters(), lr=self.lr_assign)
                
                init_iterator = tqdm(range(self.assign_init_epochs), desc="AssignInit") if self.verbose else range(self.assign_init_epochs)
                
                for init_epoch in init_iterator:
                    opt_init.zero_grad()
                    
                    # Get logits (before softmax)
                    if self.assignment_type == "nn":
                        features = X_t
                    else:  # "nn_dist"
                        distances = torch.sqrt(((X_t.unsqueeze(1) - self.centers_.unsqueeze(0)) ** 2).sum(dim=2) + 1e-8)
                        features = torch.cat([X_t, distances], dim=1)
                    
                    logits = self.assignment_net_.net(features)  # (n, K) - raw logits
                    loss = ce_loss(logits, km_labels_t)
                    loss.backward()
                    opt_init.step()
                
                if self.verbose:
                    # Check accuracy after pre-training
                    with torch.no_grad():
                        if self.assignment_type == "nn":
                            A_init = self.assignment_net_(X_t, None)
                        else:
                            A_init = self.assignment_net_(X_t, self.centers_)
                        pred_labels = A_init.argmax(dim=1).cpu().numpy()
                        accuracy = (pred_labels == km_labels).mean()
                    print(f"[DRSFC] Assignment init complete. Accuracy vs K-means: {accuracy:.4f}")
        else:  # "distance"
            self.assignment_net_ = None
            # For "distance" mode, tau is also fixed (stored as self.tau)
        
        self.discriminator_ = Discriminator(K, self.hidden_dim).to(self.device)
        # v 초기화: uniform (1/M) -> sphere 위로 정규화
        # lr_v=0이면 v가 고정됨 (김건웅님 제안)
        self.v_ = nn.Parameter(torch.ones(self.M_, device=self.device) / self.M_)
        with torch.no_grad():
            self.v_.data = self.v_.data / (self.v_.data.norm() + 1e-8)
        
        # Optimizers setup based on center_update and assignment_type
        if self.assignment_type in ("nn", "nn_dist"):
            main_params = list(self.assignment_net_.parameters())
        else:
            # "distance" mode: no learnable params for assignment (tau is fixed)
            main_params = []
        
        if self.center_update == "sgd":
            main_params = main_params + [self.centers_]
        
        opt_main = optim.Adam(main_params, lr=self.lr_assign)
        opt_disc = optim.Adam(self.discriminator_.parameters(), lr=self.lr_disc)
        opt_v = optim.Adam([self.v_], lr=self.lr_v)
        
        self.history_ = {
            "cluster_loss": [], 
            "size_loss": [],  # Cluster size regularization loss
            "fair_loss": [],  # Kept for backward compatibility (same as fair_loss_outer)
            "fair_loss_inner": [],  # Discriminator's perspective (after inner loop)
            "fair_loss_outer": [],  # Assignment's perspective (during outer loop)
            "total_loss": [],
            "centers": [],
            "soft_assignments": [],
            # Actual metrics per epoch (computed from hard assignments)
            "cost_per_epoch": [],
            "balance_per_epoch": [],
            # Discriminator tracking
            "disc_outputs": [],  # (epochs, n) - g(A) outputs
            "v": [],  # (epochs, M) - v vector
            # Assignment analysis
            "cluster_sizes": [],  # (epochs, K) - cluster sizes per epoch
            # Gradient norms
            "grad_norm_assign": [],  # Assignment network gradient norm
            "grad_norm_disc": [],  # Discriminator gradient norm
            "grad_norm_v": [],  # v gradient norm
        }
        
        # ========== WARMUP PHASE ==========
        # Train discriminator and v only (assignment network frozen)
        if self.warmup_epochs > 0 and self.lambda_fair > 0 and self.M_ > 0:
            if self.verbose:
                print(f"[DRSFC] Warmup phase: {self.warmup_epochs} epochs (discriminator only)")
            
            warmup_iterator = tqdm(range(self.warmup_epochs), desc="Warmup") if self.verbose else range(self.warmup_epochs)
            
            for warmup_epoch in warmup_iterator:
                # Compute current assignment (frozen)
                with torch.no_grad():
                    if self.assignment_type == "nn":
                        A = self.assignment_net_(X_t, None)
                    elif self.assignment_type == "nn_dist":
                        A = self.assignment_net_(X_t, self.centers_)
                    else:  # "distance"
                        sq_dist = ((X_t.unsqueeze(1) - self.centers_.unsqueeze(0)) ** 2).sum(dim=2)
                        A = torch.softmax(-sq_dist / self.tau, dim=1)
                
                # Train discriminator and v
                for _ in range(self.n_disc_steps):
                    opt_disc.zero_grad()
                    opt_v.zero_grad()
                    L_fair = fairness_loss(A, C, self.v_, self.discriminator_)
                    L_fair.backward()
                    # Apply gradient clipping to discriminator if specified
                    if self.grad_clip_disc is not None:
                        torch.nn.utils.clip_grad_norm_(self.discriminator_.parameters(), self.grad_clip_disc)
                    # Apply gradient clipping to v if specified
                    if self.grad_clip_v is not None and self.v_.grad is not None:
                        torch.nn.utils.clip_grad_norm_([self.v_], self.grad_clip_v)
                    opt_disc.step()
                    opt_v.step()
                    with torch.no_grad():
                        self.v_.data = self.v_.data / (self.v_.data.norm() + 1e-8)
            
            if self.verbose:
                with torch.no_grad():
                    A = self.assignment_net_(X_t, None) if self.assignment_type == "nn" else \
                        self.assignment_net_(X_t, self.centers_) if self.assignment_type == "nn_dist" else \
                        torch.softmax(-((X_t.unsqueeze(1) - self.centers_.unsqueeze(0)) ** 2).sum(dim=2) / self.tau, dim=1)
                    L_fair_after_warmup = fairness_loss(A, C, self.v_, self.discriminator_).item()
                print(f"[DRSFC] Warmup complete. Fair Loss after warmup: {L_fair_after_warmup:.4f}")
        
        # ========== MAIN TRAINING LOOP ==========
        iterator = tqdm(range(self.max_iter), desc="DRSFC") if self.verbose else range(self.max_iter)
        
        def compute_A(X_t):
            """Compute soft assignments based on assignment_type.
            
            Args:
                X_t: (n, d) data points tensor
            
            Returns:
                A: (n, K) soft assignment matrix
            """
            if self.assignment_type == "nn":
                A = self.assignment_net_(X_t, None)
            elif self.assignment_type == "nn_dist":
                A = self.assignment_net_(X_t, self.centers_)
            else:  # "distance"
                sq_dist = ((X_t.unsqueeze(1) - self.centers_.unsqueeze(0)) ** 2).sum(dim=2)
                A = torch.softmax(-sq_dist / self.tau, dim=1)
            
            return A 
        
        def compute_grad_norm(parameters):
            """Compute total gradient norm for a set of parameters."""
            total_norm = 0.0
            for p in parameters:
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            return total_norm ** 0.5
        
        for epoch in iterator:
            # Inner loop: maximize L_fair (discriminator learns to predict A from subgroups)
            L_fair_inner = torch.tensor(0.0)
            grad_norm_disc_epoch = 0.0
            grad_norm_v_epoch = 0.0
            if self.lambda_fair > 0 and self.M_ > 0:
                for _ in range(self.n_disc_steps):
                    opt_disc.zero_grad()
                    opt_v.zero_grad()
                    with torch.no_grad():
                        A = compute_A(X_t)
                    L_fair_inner = fairness_loss(A, C, self.v_, self.discriminator_)
                    L_fair_inner.backward()
                    # Record gradient norms (last step of inner loop)
                    grad_norm_disc_epoch = compute_grad_norm(self.discriminator_.parameters())
                    grad_norm_v_epoch = self.v_.grad.norm().item() if self.v_.grad is not None else 0.0
                    # Apply gradient clipping to discriminator if specified
                    if self.grad_clip_disc is not None:
                        torch.nn.utils.clip_grad_norm_(self.discriminator_.parameters(), self.grad_clip_disc)
                    # Apply gradient clipping to v if specified
                    if self.grad_clip_v is not None and self.v_.grad is not None:
                        torch.nn.utils.clip_grad_norm_([self.v_], self.grad_clip_v)
                    opt_disc.step()
                    opt_v.step()
                    with torch.no_grad():
                        self.v_.data = self.v_.data / (self.v_.data.norm() + 1e-8)
            
            # Outer loop: minimize L_cluster + lambda_size * L_size - lambda_fair * L_fair
            opt_main.zero_grad()
            A = compute_A(X_t)
            L_cluster = clustering_loss(A, X_t, self.centers_)
            
            # Cluster size regularization: L_size = sum_k max(0, epsilon - p_k)
            L_size = torch.tensor(0.0, device=self.device)
            if self.epsilon_size is not None:
                eps_val = None if self.epsilon_size == 'auto' else self.epsilon_size
                L_size = cluster_size_loss(A, eps_val)
            
            if self.lambda_fair > 0 and self.M_ > 0:
                self.discriminator_.eval()
                for p in self.discriminator_.parameters():
                    p.requires_grad_(False)
                L_fair_outer = fairness_loss(A, C, self.v_.detach(), self.discriminator_)
                self.discriminator_.train()
                for p in self.discriminator_.parameters():
                    p.requires_grad_(True)
                total_loss = L_cluster + self.lambda_size * L_size - self.lambda_fair * L_fair_outer
            else:
                L_fair_outer = torch.tensor(0.0)
                total_loss = L_cluster + self.lambda_size * L_size
            
            total_loss.backward()
            
            # Record assignment gradient norm before step
            grad_norm_assign_epoch = 0.0
            if self.assignment_type in ("nn", "nn_dist") and self.assignment_net_ is not None:
                grad_norm_assign_epoch = compute_grad_norm(self.assignment_net_.parameters())
            
            # Apply gradient clipping to assignment network if specified
            if self.grad_clip_assign is not None and self.assignment_net_ is not None:
                torch.nn.utils.clip_grad_norm_(self.assignment_net_.parameters(), self.grad_clip_assign)
            
            opt_main.step()
            
            # M-step: update centers with closed-form solution (only if center_update=="mstep")
            # If center_update=="fixed", centers remain at initial KMeans solution
            if self.center_update == "mstep":
                with torch.no_grad():
                    A_detached = compute_A(X_t)  # (n, K)
                    sum_A = A_detached.sum(dim=0)  # (K,)
                    eps = 1e-8
                    weighted_X = A_detached.T @ X_t  # (K, d)
                    self.centers_.data = weighted_X / (sum_A.unsqueeze(1) + eps)
            # elif self.center_update == "fixed": pass (do nothing, centers stay fixed)
            
            # Record losses
            self.history_["cluster_loss"].append(L_cluster.item())
            size_loss_val = L_size.item() if isinstance(L_size, torch.Tensor) else L_size
            self.history_["size_loss"].append(size_loss_val)
            fair_inner_val = L_fair_inner.item() if isinstance(L_fair_inner, torch.Tensor) else L_fair_inner
            fair_outer_val = L_fair_outer.item() if isinstance(L_fair_outer, torch.Tensor) else L_fair_outer
            self.history_["fair_loss_inner"].append(fair_inner_val)
            self.history_["fair_loss_outer"].append(fair_outer_val)
            self.history_["fair_loss"].append(fair_outer_val)  # Backward compatibility
            self.history_["total_loss"].append(total_loss.item())
            
            # Record gradient norms
            self.history_["grad_norm_assign"].append(grad_norm_assign_epoch)
            self.history_["grad_norm_disc"].append(grad_norm_disc_epoch)
            self.history_["grad_norm_v"].append(grad_norm_v_epoch)
            
            # Track centers and soft assignments per epoch
            with torch.no_grad():
                A_epoch = compute_A(X_t)
                self.history_["centers"].append(self.centers_.detach().cpu().numpy().copy())
                self.history_["soft_assignments"].append(A_epoch.cpu().numpy().copy())
                
                # Track discriminator outputs and v
                disc_out = self.discriminator_(A_epoch).cpu().numpy().copy()
                self.history_["disc_outputs"].append(disc_out)
                self.history_["v"].append(self.v_.detach().cpu().numpy().copy())
                
                # Track cluster sizes
                labels_epoch = A_epoch.argmax(dim=1).cpu().numpy()
                cluster_sizes = np.array([np.sum(labels_epoch == k) for k in range(K)])
                self.history_["cluster_sizes"].append(cluster_sizes)
                
                # Compute actual metrics per epoch using evaluation.py functions
                if track_metrics:
                    # Cost: SSE / n (using evaluation.py's compute_cost)
                    cost = compute_cost(X_np, labels_epoch, K)
                    self.history_["cost_per_epoch"].append(cost)
                    
                    # Balance: subgroup balance metric (using evaluation.py's compute_subgroup_balance)
                    balance = self._compute_balance(labels_epoch, S_np, K)
                    self.history_["balance_per_epoch"].append(balance)
        
        with torch.no_grad():
            A = compute_A(X_t)
            self.labels_ = A.argmax(dim=1).cpu().numpy()
            self.assignment_probs_ = A.cpu().numpy()
        
        if self.verbose:
            print("[DRSFC] Done!")
        
        return self
    
    def _compute_balance(self, labels, S, K):
        """
        Compute subgroup balance metric using evaluation.py's compute_subgroup_balance.
        
        Higher is better (1.0 = perfectly balanced).
        """
        result = compute_subgroup_balance(S, labels, K)
        return result['min_balance']
    
    def predict(self, X):
        """Predict cluster labels for new data."""
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if self.assignment_type == "nn":
                A = self.assignment_net_(X_t, None)
            elif self.assignment_type == "nn_dist":
                A = self.assignment_net_(X_t, self.centers_)
            else:  # "distance"
                sq_dist = ((X_t.unsqueeze(1) - self.centers_.unsqueeze(0)) ** 2).sum(dim=2)
                A = torch.softmax(-sq_dist / self.tau, dim=1)
        return A.argmax(dim=1).cpu().numpy()
    
    def predict_proba(self, X):
        """Predict soft assignment probabilities for new data."""
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if self.assignment_type == "nn":
                A = self.assignment_net_(X_t, None)
            elif self.assignment_type == "nn_dist":
                A = self.assignment_net_(X_t, self.centers_)
            else:  # "distance"
                sq_dist = ((X_t.unsqueeze(1) - self.centers_.unsqueeze(0)) ** 2).sum(dim=2)
                A = torch.softmax(-sq_dist / self.tau, dim=1)
        return A.cpu().numpy()
    
    def get_centers(self):
        """Get cluster centers."""
        return self.centers_.detach().cpu().numpy()


# Runners
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
    # Default: use GPU when available. You can force CPU with --cpu.
    force_cpu = bool(getattr(args, 'cpu', False))
    use_cuda = (not force_cpu) and torch.cuda.is_available()
    # Backward compatibility: if user explicitly passed --use_cuda, honor it.
    if getattr(args, 'use_cuda', False):
        use_cuda = True and (not force_cpu) and torch.cuda.is_available()
    device = get_device() if use_cuda else 'cpu'
    print(f"\n[2/4] Device: {device}")
    
    # Unfair baseline
    print("\n[Unfair] K-Means baseline...")
    km = KMeans(n_clusters=K, random_state=args.seed, n_init=10)
    unfair_labels = km.fit_predict(X)
    unfair_metrics = compute_all(X, y, S, unfair_labels, K)
    print(f"[Unfair] Cost / Balance: {unfair_metrics['Cost']:.4f} / {unfair_metrics['SubgroupBalance']:.4f}")
    
    # DRSFC
    print("\n[3/4] Training DRSFC...")

    init_centers = None
    init_centers_path = getattr(args, 'init_centers_path', None)
    if init_centers_path:
        init_centers = np.load(init_centers_path)

    # Parse epsilon_size: can be None, 'auto', or a float string
    epsilon_size_arg = getattr(args, 'epsilon_size', None)
    if epsilon_size_arg is None or epsilon_size_arg == 'None':
        epsilon_size = None
    elif epsilon_size_arg == 'auto':
        epsilon_size = 'auto'
    else:
        epsilon_size = float(epsilon_size_arg)

    model = DRSFC(
        n_clusters=K,
        lambda_fair=args.lambda_fair,
        gamma=args.gamma,
        max_order=args.max_order,
        lr=args.lr,
        lr_assign=getattr(args, 'lr_assign', None),
        lr_disc=getattr(args, 'lr_disc', None),
        lr_v=getattr(args, 'lr_v', None),
        n_disc_steps=args.n_disc_steps,
        max_iter=args.epochs,
        verbose=args.verbose,
        random_state=args.seed,
        device=device,
        center_update=getattr(args, 'center_update', 'mstep'),
        assignment_type=getattr(args, 'assignment_type', 'nn'),
        center_init=getattr(args, 'center_init', 'k-means++'),
        init_centers=init_centers,
        grad_clip_disc=getattr(args, 'grad_clip_disc', None),
        epsilon_size=epsilon_size,
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
        'lr_assign': getattr(args, 'lr_assign', None),
        'lr_disc': getattr(args, 'lr_disc', None),
        'lr_v': getattr(args, 'lr_v', None),
        'grad_clip_disc': getattr(args, 'grad_clip_disc', None),
        'epsilon_size': epsilon_size,
        # Keep key name for backward compatibility with existing scripts
        'max_iter': args.epochs,
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
    
    # Data (for reproducibility)
    np.save(save_dir / "X.npy", X)
    np.save(save_dir / "y.npy", y)
    np.save(save_dir / "S.npy", S.values if hasattr(S, 'values') else S)
    
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
    
    # Data (for reproducibility)
    np.save(save_dir / "X.npy", X)
    np.save(save_dir / "y.npy", y)
    np.save(save_dir / "S.npy", S.values if hasattr(S, 'values') else S)
    
    print(f"\nResults saved to: {save_dir}/")
    print(f"\n[Result] Cost / Balance: {metrics['Cost']:.4f} / {metrics['SubgroupBalance']:.4f}")
