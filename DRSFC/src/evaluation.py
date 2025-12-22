"""
Evaluation metrics for DRSFC.

Metrics:
- Cost: SSE / n (clustering quality)
- Balance: min pairwise balance across clusters
- MarginalBalance: per-attribute balance
- SubgroupBalance: full subgroup balance
- Delta: max deviation from global proportions
- MP(l): l-th order Marginal Parity
- SP: Subgroup Parity
- WMP: Wasserstein Marginal Parity (soft clustering)
"""

import numpy as np
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from sklearn import metrics as sklearn_metrics
from itertools import combinations, product
from collections import Counter
from typing import Dict, List, Optional, Tuple

try:
    from munkres import Munkres
    MUNKRES_AVAILABLE = True
except ImportError:
    MUNKRES_AVAILABLE = False


def evaluation(color_xs: List[np.ndarray], 
               color_assignments: List[np.ndarray], 
               centers: np.ndarray, 
               K: int) -> Tuple[float, float]:
    """
    Evaluates clustering results (FCA-style).
    
    Args:
        color_xs: List of feature arrays per color/group
        color_assignments: List of cluster assignments per color
        centers: Cluster centers (K, d)
        K: Number of clusters
    
    Returns:
        (cost, balance): Tuple of cost and balance metrics
    """
    n = sum(xs.shape[0] for xs in color_xs)
    
    # Compute cost
    objective = 0.0
    cluster_cnts = []
    
    for xs_i, assignments_i in zip(color_xs, color_assignments):
        distances_i = distance.cdist(xs_i, centers, metric='minkowski', p=2)
        objective += (distances_i[np.arange(len(assignments_i)), assignments_i]**2).sum()
        
        sub_cluster_cnts = []
        for k in range(K):
            sub_cluster_cnts.append((assignments_i == k).sum())
        cluster_cnts.append(sub_cluster_cnts)
    
    # Compute balance (min ratio across colors for each cluster)
    cluster_cnts = np.array(cluster_cnts)
    n_colors = len(color_xs)
    
    min_ratios = []
    for k in range(K):
        for i in range(n_colors):
            for j in range(i+1, n_colors):
                a_k = cluster_cnts[i, k]
                b_k = cluster_cnts[j, k]
                if a_k == 0 or b_k == 0:
                    min_ratios.append(0.0)
                else:
                    min_ratios.append(min(a_k / b_k, b_k / a_k))
    
    balance = min(min_ratios) if min_ratios else 0.0
    
    return objective / n, balance


def compute_cost(X: np.ndarray, cluster_ids: np.ndarray, K: int) -> float:
    """SSE / n"""
    n = X.shape[0]
    cost = 0.0
    for k in range(K):
        mask = cluster_ids == k
        if mask.any():
            center = X[mask].mean(axis=0)
            cost += ((X[mask] - center)**2).sum()
    return float(cost / n)


def compute_acc(y_true: np.ndarray, cluster_ids: np.ndarray) -> float:
    """Hungarian ACC"""
    if not MUNKRES_AVAILABLE:
        C = sklearn_metrics.confusion_matrix(y_true, cluster_ids)
        return float(np.sum(C.max(axis=0)) / C.sum())
    
    labels_true = np.unique(y_true)
    labels_pred = np.unique(cluster_ids)
    K = max(len(labels_true), len(labels_pred))
    
    C = np.zeros((len(labels_true), len(labels_pred)), dtype=int)
    for i, t in enumerate(labels_true):
        for j, p in enumerate(labels_pred):
            C[i, j] = int(np.sum((y_true == t) & (cluster_ids == p)))
    
    C_pad = np.zeros((K, K), dtype=int)
    C_pad[:C.shape[0], :C.shape[1]] = C
    cost_mat = np.array([[C_pad[:, j].sum() - C_pad[i, j] for i in range(K)] for j in range(K)])
    mapping = dict(Munkres().compute(cost_mat.tolist()))
    
    mapped = []
    for c in cluster_ids:
        j = int(np.where(labels_pred == c)[0][0])
        i = mapping.get(j, 0)
        mapped.append(labels_true[i] if i < len(labels_true) else labels_true[0])
    
    return float((np.array(mapped) == y_true).mean())


def compute_marginal_balance(S: np.ndarray, cluster_ids: np.ndarray, K: int) -> Dict[str, float]:
    """Compute balance for each sensitive attribute separately."""
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    
    q = S.shape[1]
    marginal_balances = {}
    
    for attr_idx in range(q):
        attr_vals = S[:, attr_idx]
        unique_vals = np.unique(attr_vals)
        
        if len(unique_vals) < 2:
            marginal_balances[f'attr_{attr_idx}'] = 1.0
            continue
        
        attr_balances = []
        for k in range(K):
            mask = cluster_ids == k
            counts = {v: (mask & (attr_vals == v)).sum() for v in unique_vals}
            
            for i, j in combinations(unique_vals, 2):
                if counts[i] == 0 or counts[j] == 0:
                    attr_balances.append(0.0)
                else:
                    attr_balances.append(min(counts[i]/counts[j], counts[j]/counts[i]))
        
        marginal_balances[f'attr_{attr_idx}'] = float(min(attr_balances)) if attr_balances else 0.0
    
    marginal_balances['min'] = min(marginal_balances.values()) if marginal_balances else 0.0
    return marginal_balances


def compute_subgroup_balance(S: np.ndarray, cluster_ids: np.ndarray, K: int) -> Dict[str, float]:
    """Compute subgroup balance (full intersection of all sensitive attributes)."""
    if S.ndim == 1:
        subgroups = np.array([str(s) for s in S])
    else:
        subgroups = np.array([''.join(map(str, row)) for row in S])
    
    unique_subgroups = np.unique(subgroups)
    
    if len(unique_subgroups) < 2:
        return {'min_balance': 1.0, 'avg_balance': 1.0, 'n_subgroups': len(unique_subgroups)}
    
    cluster_balances = []
    for k in range(K):
        mask = cluster_ids == k
        cluster_sg = subgroups[mask]
        counts = Counter(cluster_sg)
        
        pair_balances = []
        for i, j in combinations(unique_subgroups, 2):
            ci, cj = counts.get(i, 0), counts.get(j, 0)
            if ci == 0 or cj == 0:
                pair_balances.append(0.0)
            else:
                pair_balances.append(min(ci/cj, cj/ci))
        
        cluster_balances.append(min(pair_balances) if pair_balances else 0.0)
    
    return {
        'min_balance': float(min(cluster_balances)),
        'avg_balance': float(np.mean(cluster_balances)),
        'n_subgroups': len(unique_subgroups),
    }


def compute_delta(S: np.ndarray, cluster_ids: np.ndarray, K: int) -> float:
    """Compute Delta (max gap from global proportions)."""
    if S.ndim == 1:
        subgroups = np.array([str(s) for s in S])
    else:
        subgroups = np.array([''.join(map(str, row)) for row in S])
    
    unique_subgroups = np.unique(subgroups)
    n = len(subgroups)
    
    global_props = {sg: np.sum(subgroups == sg) / n for sg in unique_subgroups}
    
    max_delta = 0.0
    for k in range(K):
        mask = cluster_ids == k
        n_k = mask.sum()
        if n_k == 0:
            continue
        
        cluster_sg = subgroups[mask]
        for sg in unique_subgroups:
            cluster_prop = np.sum(cluster_sg == sg) / n_k
            delta = abs(cluster_prop - global_props[sg])
            max_delta = max(max_delta, delta)
    
    return float(max_delta)


def compute_MP(S: np.ndarray, cluster_ids: np.ndarray, K: int, order: int = 1) -> float:
    """Compute MP(l) - l-th order Marginal Parity."""
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    
    n, q = S.shape
    if order > q:
        return 0.0
    
    mp_values = []
    
    for k in range(K):
        cluster_mask = cluster_ids == k
        n_k = cluster_mask.sum()
        p_hat = n_k / n
        
        for L in combinations(range(q), order):
            L = list(L)
            S_L = S[:, L]
            unique_vals = [np.unique(S[:, j]) for j in L]
            
            for a in product(*unique_vals):
                mask_a = np.all(S_L == np.array(a), axis=1)
                n_L_a = mask_a.sum()
                
                if n_L_a == 0:
                    continue
                
                n_L_a_k = (mask_a & cluster_mask).sum()
                p_L_a = n_L_a_k / n_L_a
                
                mp_values.append((n_L_a / n) * abs(p_L_a - p_hat))
    
    return float(sum(mp_values)) if mp_values else 0.0


def compute_SP(S: np.ndarray, cluster_ids: np.ndarray, K: int) -> float:
    """Compute SP - Subgroup Parity."""
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    
    n = S.shape[0]
    subgroups = np.array([''.join(map(str, row)) for row in S])
    unique_subgroups = np.unique(subgroups)
    
    sp_max = 0.0
    
    for k in range(K):
        cluster_mask = cluster_ids == k
        p_hat = cluster_mask.sum() / n
        
        for s in unique_subgroups:
            mask_s = subgroups == s
            n_s = mask_s.sum()
            
            if n_s == 0:
                continue
            
            n_s_k = (mask_s & cluster_mask).sum()
            p_s = n_s_k / n_s
            
            sp_val = (n_s / n) * abs(p_s - p_hat)
            sp_max = max(sp_max, sp_val)
    
    return float(sp_max)


def compute_WMP(S: np.ndarray, soft_assignments: np.ndarray, K: int) -> Optional[float]:
    """Compute WMP - Wasserstein Marginal Parity."""
    if soft_assignments is None:
        return None
    
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    
    n, q = S.shape
    f_all = np.sum(soft_assignments * np.arange(K), axis=1)
    
    wmp_max = 0.0
    
    for j in range(q):
        attr_j = S[:, j]
        
        for a in np.unique(attr_j):
            mask_a = attr_j == a
            n_j_a = mask_a.sum()
            
            if n_j_a == 0:
                continue
            
            f_a = f_all[mask_a]
            w1 = wasserstein_distance(f_a, f_all)
            wmp_val = (n_j_a / n) * w1
            wmp_max = max(wmp_max, wmp_val)
    
    return float(wmp_max)


def compute_all(X: np.ndarray, y: np.ndarray, S: np.ndarray, 
                cluster_ids: np.ndarray, K: int,
                soft_assignments: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute all metrics."""
    marginal = compute_marginal_balance(S, cluster_ids, K)
    subgroup = compute_subgroup_balance(S, cluster_ids, K)
    
    metrics = {
        'Cost': round(compute_cost(X, cluster_ids, K), 4),
        'Acc': round(compute_acc(y, cluster_ids), 4),
        'ARI': round(sklearn_metrics.adjusted_rand_score(y, cluster_ids), 4),
        'NMI': round(sklearn_metrics.normalized_mutual_info_score(y, cluster_ids), 4),
        'Delta': round(compute_delta(S, cluster_ids, K), 4),
        'MarginalBalance': round(marginal['min'], 4),
        'SubgroupBalance': round(subgroup['min_balance'], 4),
        'MP1': round(compute_MP(S, cluster_ids, K, order=1), 4),
        'MP2': round(compute_MP(S, cluster_ids, K, order=2), 4),
        'SP': round(compute_SP(S, cluster_ids, K), 4),
    }
    
    if soft_assignments is not None:
        wmp = compute_WMP(S, soft_assignments, K)
        metrics['WMP'] = round(wmp, 4) if wmp is not None else None
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Results") -> None:
    """Print metrics in formatted table."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"  Cost:            {metrics['Cost']:.4f}")
    print(f"  Balance:         {metrics['SubgroupBalance']:.4f}")
    print(f"  MarginalBalance: {metrics['MarginalBalance']:.4f}")
    print(f"  Delta:           {metrics['Delta']:.4f}")
    print(f"  MP(1):           {metrics['MP1']:.4f}")
    print(f"  MP(2):           {metrics['MP2']:.4f}")
    print(f"  SP:              {metrics['SP']:.4f}")
    if 'WMP' in metrics and metrics['WMP'] is not None:
        print(f"  WMP:             {metrics['WMP']:.4f}")
    print(f"  Acc:             {metrics['Acc']:.4f}")
    print(f"  ARI:             {metrics['ARI']:.4f}")
    print(f"  NMI:             {metrics['NMI']:.4f}")
    print(f"{'='*60}\n")
