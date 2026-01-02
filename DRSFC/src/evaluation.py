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


def compute_marginal_balance(S: np.ndarray, cluster_ids: np.ndarray, K: int,
                              soft_assignments: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute balance for each sensitive attribute separately (hard and soft versions).
    
    Hard: n_{j,a,k}^{hard} = sum_i I(S_{ij}=a) * I(A(x_i)=k)
    Soft: n_{j,a,k}^{soft} = sum_i I(S_{ij}=a) * A(x_i)_k
    """
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    
    q = S.shape[1]
    marginal_balances = {}
    
    # === Hard Marginal Balance ===
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
    
    # === Soft Marginal Balance ===
    if soft_assignments is not None:
        for attr_idx in range(q):
            attr_vals = S[:, attr_idx]
            unique_vals = np.unique(attr_vals)
            
            if len(unique_vals) < 2:
                marginal_balances[f'attr_{attr_idx}(soft)'] = 1.0
                continue
            
            attr_balances_soft = []
            for k in range(K):
                # Soft counts: sum of soft assignment probabilities for each attribute value
                counts_soft = {}
                for v in unique_vals:
                    mask_v = attr_vals == v
                    counts_soft[v] = soft_assignments[mask_v, k].sum()
                
                for i, j in combinations(unique_vals, 2):
                    if counts_soft[i] < 1e-10 or counts_soft[j] < 1e-10:
                        attr_balances_soft.append(0.0)
                    else:
                        attr_balances_soft.append(min(counts_soft[i]/counts_soft[j], counts_soft[j]/counts_soft[i]))
            
            marginal_balances[f'attr_{attr_idx}(soft)'] = float(min(attr_balances_soft)) if attr_balances_soft else 0.0
        
        soft_attrs = [v for k, v in marginal_balances.items() if k.endswith('(soft)')]
        marginal_balances['min(soft)'] = min(soft_attrs) if soft_attrs else 0.0
    
    return marginal_balances


def compute_subgroup_balance(S: np.ndarray, cluster_ids: np.ndarray, K: int,
                              soft_assignments: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute subgroup balance (full intersection of all sensitive attributes).
    
    Hard: n_{s,k}^{hard} = sum_i I(S_i=s) * I(A(x_i)=k)
    Soft: n_{s,k}^{soft} = sum_i I(S_i=s) * A(x_i)_k
    
    Balance ratio: min(n_s/n_s', n_s'/n_s) for each subgroup pair (s, s') in each cluster k.
    """
    if S.ndim == 1:
        subgroups = np.array([str(s) for s in S])
    else:
        subgroups = np.array([''.join(map(str, row)) for row in S])
    
    unique_subgroups = np.unique(subgroups)
    
    if len(unique_subgroups) < 2:
        result = {'min_balance': 1.0, 'avg_balance': 1.0, 'n_subgroups': len(unique_subgroups)}
        if soft_assignments is not None:
            result['min_balance(soft)'] = 1.0
            result['avg_balance(soft)'] = 1.0
        return result
    
    # === Hard Subgroup Balance ===
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
    
    result = {
        'min_balance': float(min(cluster_balances)),
        'avg_balance': float(np.mean(cluster_balances)),
        'n_subgroups': len(unique_subgroups),
    }
    
    # === Soft Subgroup Balance ===
    if soft_assignments is not None:
        cluster_balances_soft = []
        for k in range(K):
            # Soft counts for each subgroup in cluster k
            counts_soft = {}
            for sg in unique_subgroups:
                mask_sg = subgroups == sg
                counts_soft[sg] = soft_assignments[mask_sg, k].sum()
            
            pair_balances_soft = []
            for i, j in combinations(unique_subgroups, 2):
                ci, cj = counts_soft[i], counts_soft[j]
                if ci < 1e-10 or cj < 1e-10:
                    pair_balances_soft.append(0.0)
                else:
                    pair_balances_soft.append(min(ci/cj, cj/ci))
            
            cluster_balances_soft.append(min(pair_balances_soft) if pair_balances_soft else 0.0)
        
        result['min_balance(soft)'] = float(min(cluster_balances_soft))
        result['avg_balance(soft)'] = float(np.mean(cluster_balances_soft))
    
    return result


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


def compute_gap(S: np.ndarray, cluster_ids: np.ndarray, K: int,
                soft_assignments: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute Gap (Δ) - Fairness Gap between subgroups.
    
    Compare cluster distributions between all subgroup pairs (intersection of all attributes).
    For subgroups s, s':
        Δ_{s,s'} = (1/2) * sum_k | p_k^{(s)} - p_k^{(s')} |
    
    This is half of the Total Variation distance between two group distributions.
    
    Aggregations:
        _sum: sum of all pairwise gaps
        _max: maximum gap (worst-case unfairness)
        _avg: average gap
        _wsum: weighted sum (weight = min(n_s, n_s') / n)
        _wmax: weighted max
    
    Note: Subgroups are defined as the full intersection of all sensitive attributes.
          For S with q attributes, this creates 2^q possible subgroups (if binary).
    """
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    
    n, q = S.shape
    results = {}
    
    # ========== SUBGROUP GAP (full intersection) ==========
    subgroups = np.array([''.join(map(str, row)) for row in S])
    unique_subgroups = np.unique(subgroups)
    
    if len(unique_subgroups) >= 2:
        # --- Hard Subgroup Gap ---
        sg_props = {}
        sg_counts = {}
        for sg in unique_subgroups:
            mask_sg = subgroups == sg
            n_sg = mask_sg.sum()
            sg_counts[sg] = n_sg
            if n_sg == 0:
                sg_props[sg] = np.zeros(K)
            else:
                sg_props[sg] = np.array([(mask_sg & (cluster_ids == k)).sum() / n_sg for k in range(K)])
        
        sgap_values_hard = []
        sgap_weights_hard = []
        for i, sg1 in enumerate(unique_subgroups):
            for sg2 in unique_subgroups[i+1:]:
                gap = 0.5 * np.sum(np.abs(sg_props[sg1] - sg_props[sg2]))
                weight = min(sg_counts[sg1], sg_counts[sg2]) / n
                sgap_values_hard.append(gap)
                sgap_weights_hard.append(weight)
        
        if sgap_values_hard:
            results['Gap_sum(hard)'] = float(sum(sgap_values_hard))
            results['Gap_max(hard)'] = float(max(sgap_values_hard))
            results['Gap_avg(hard)'] = float(np.mean(sgap_values_hard))
            results['Gap_wsum(hard)'] = float(sum(g * w for g, w in zip(sgap_values_hard, sgap_weights_hard)))
            results['Gap_wmax(hard)'] = float(max(g * w for g, w in zip(sgap_values_hard, sgap_weights_hard)))
        else:
            for suffix in ['sum', 'max', 'avg', 'wsum', 'wmax']:
                results[f'Gap_{suffix}(hard)'] = 0.0
        
        # --- Soft Subgroup Gap ---
        if soft_assignments is not None:
            sg_props_soft = {}
            for sg in unique_subgroups:
                mask_sg = subgroups == sg
                n_sg = mask_sg.sum()
                if n_sg == 0:
                    sg_props_soft[sg] = np.zeros(K)
                else:
                    sg_props_soft[sg] = soft_assignments[mask_sg].mean(axis=0)
            
            sgap_values_soft = []
            sgap_weights_soft = []
            for i, sg1 in enumerate(unique_subgroups):
                for sg2 in unique_subgroups[i+1:]:
                    gap = 0.5 * np.sum(np.abs(sg_props_soft[sg1] - sg_props_soft[sg2]))
                    weight = min(sg_counts[sg1], sg_counts[sg2]) / n
                    sgap_values_soft.append(gap)
                    sgap_weights_soft.append(weight)
            
            if sgap_values_soft:
                results['Gap_sum(soft)'] = float(sum(sgap_values_soft))
                results['Gap_max(soft)'] = float(max(sgap_values_soft))
                results['Gap_avg(soft)'] = float(np.mean(sgap_values_soft))
                results['Gap_wsum(soft)'] = float(sum(g * w for g, w in zip(sgap_values_soft, sgap_weights_soft)))
                results['Gap_wmax(soft)'] = float(max(g * w for g, w in zip(sgap_values_soft, sgap_weights_soft)))
            else:
                for suffix in ['sum', 'max', 'avg', 'wsum', 'wmax']:
                    results[f'Gap_{suffix}(soft)'] = 0.0
    else:
        for suffix in ['sum', 'max', 'avg', 'wsum', 'wmax']:
            results[f'Gap_{suffix}(hard)'] = 0.0
            if soft_assignments is not None:
                results[f'Gap_{suffix}(soft)'] = 0.0
    
    return results
    
    if len(unique_subgroups) >= 2:
        # --- Hard Subgroup Gap ---
        sg_props = {}
        sg_counts = {}
        for sg in unique_subgroups:
            mask_sg = subgroups == sg
            n_sg = mask_sg.sum()
            sg_counts[sg] = n_sg
            if n_sg == 0:
                sg_props[sg] = np.zeros(K)
            else:
                sg_props[sg] = np.array([(mask_sg & (cluster_ids == k)).sum() / n_sg for k in range(K)])
        
        sgap_values_hard = []
        sgap_weights_hard = []
        for i, sg1 in enumerate(unique_subgroups):
            for sg2 in unique_subgroups[i+1:]:
                gap = 0.5 * np.sum(np.abs(sg_props[sg1] - sg_props[sg2]))
                weight = min(sg_counts[sg1], sg_counts[sg2]) / n
                sgap_values_hard.append(gap)
                sgap_weights_hard.append(weight)
        
        if sgap_values_hard:
            results['SubgroupGap_sum(hard)'] = float(sum(sgap_values_hard))
            results['SubgroupGap_max(hard)'] = float(max(sgap_values_hard))
            results['SubgroupGap_avg(hard)'] = float(np.mean(sgap_values_hard))
            results['SubgroupGap_wsum(hard)'] = float(sum(g * w for g, w in zip(sgap_values_hard, sgap_weights_hard)))
            results['SubgroupGap_wmax(hard)'] = float(max(g * w for g, w in zip(sgap_values_hard, sgap_weights_hard)))
        
        # --- Soft Subgroup Gap ---
        if soft_assignments is not None:
            sg_props_soft = {}
            for sg in unique_subgroups:
                mask_sg = subgroups == sg
                n_sg = mask_sg.sum()
                if n_sg == 0:
                    sg_props_soft[sg] = np.zeros(K)
                else:
                    sg_props_soft[sg] = soft_assignments[mask_sg].mean(axis=0)
            
            sgap_values_soft = []
            sgap_weights_soft = []
            for i, sg1 in enumerate(unique_subgroups):
                for sg2 in unique_subgroups[i+1:]:
                    gap = 0.5 * np.sum(np.abs(sg_props_soft[sg1] - sg_props_soft[sg2]))
                    weight = min(sg_counts[sg1], sg_counts[sg2]) / n
                    sgap_values_soft.append(gap)
                    sgap_weights_soft.append(weight)
            
            if sgap_values_soft:
                results['SubgroupGap_sum(soft)'] = float(sum(sgap_values_soft))
                results['SubgroupGap_max(soft)'] = float(max(sgap_values_soft))
                results['SubgroupGap_avg(soft)'] = float(np.mean(sgap_values_soft))
                results['SubgroupGap_wsum(soft)'] = float(sum(g * w for g, w in zip(sgap_values_soft, sgap_weights_soft)))
                results['SubgroupGap_wmax(soft)'] = float(max(g * w for g, w in zip(sgap_values_soft, sgap_weights_soft)))
    else:
        for suffix in ['sum', 'max', 'avg', 'wsum', 'wmax']:
            results[f'SubgroupGap_{suffix}(hard)'] = 0.0
            if soft_assignments is not None:
                results[f'SubgroupGap_{suffix}(soft)'] = 0.0
    
    return results
    
    return results


def compute_MP(S: np.ndarray, cluster_ids: np.ndarray, K: int, order: int = 1,
               soft_assignments: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute MP(l) - l-th order Marginal Parity (both hard and soft versions).
    
    Definitions:
        For order l, consider L ⊆ [q] such that |L| = l
        p_hat_k = global cluster k proportion
        p_hat_k_L_a = cluster k proportion among samples with s_i[L]=a
        Delta_k_L_a = |p_hat_k_L_a - p_hat_k|
        
    Aggregations (following Table 2 conventions):
        MP^(l),max  = max_{L,a,k} Delta                 (main, unweighted)
        MP^(l),wmax = max_{L,a,k} (n_L_a/n) * Delta     (main, weighted)
        MP^(l),wsum = sum_{L,a,k} (n_L_a/n) * Delta     (average deviation)
        MP^(l),avg  = (1/num_terms) * sum Delta        (normalized)
    """
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    
    n, q = S.shape
    if order > q:
        return {f'MP{order}_max(hard)': 0.0, f'MP{order}_wmax(hard)': 0.0,
                f'MP{order}_wsum(hard)': 0.0, f'MP{order}_avg(hard)': 0.0}
    
    results = {}
    
    # --- Hard MP ---
    p_hat_hard = np.zeros(K)
    for k in range(K):
        p_hat_hard[k] = (cluster_ids == k).sum() / n
    
    mp_deltas_hard = []
    mp_max_hard = 0.0
    mp_wsum_hard = 0.0
    mp_wmax_hard = 0.0
    
    for L in combinations(range(q), order):
        L = list(L)
        S_L = S[:, L]
        unique_vals = [np.unique(S[:, j]) for j in L]
        
        for a in product(*unique_vals):
            mask_a = np.all(S_L == np.array(a), axis=1)
            n_L_a = mask_a.sum()
            
            if n_L_a == 0:
                continue
            weight = n_L_a / n
            
            for k in range(K):
                n_L_a_k = (mask_a & (cluster_ids == k)).sum()
                p_L_a_k = n_L_a_k / n_L_a
                delta = abs(p_L_a_k - p_hat_hard[k])
                
                mp_deltas_hard.append(delta)
                mp_max_hard = max(mp_max_hard, delta)
                mp_wsum_hard += weight * delta
                mp_wmax_hard = max(mp_wmax_hard, weight * delta)
    
    num_terms_hard = len(mp_deltas_hard) if mp_deltas_hard else 1
    mp_sum_hard = sum(mp_deltas_hard) if mp_deltas_hard else 0.0
    mp_avg_hard = mp_sum_hard / num_terms_hard if mp_deltas_hard else 0.0
    
    results[f'MP{order}_sum(hard)'] = float(mp_sum_hard)
    results[f'MP{order}_max(hard)'] = float(mp_max_hard)
    results[f'MP{order}_avg(hard)'] = float(mp_avg_hard)
    results[f'MP{order}_wsum(hard)'] = float(mp_wsum_hard)
    results[f'MP{order}_wmax(hard)'] = float(mp_wmax_hard)
    
    # --- Soft MP ---
    if soft_assignments is not None:
        p_hat_soft = soft_assignments.mean(axis=0)  # (K,)
        
        mp_deltas_soft = []
        mp_max_soft = 0.0
        mp_wsum_soft = 0.0
        mp_wmax_soft = 0.0
        
        for L in combinations(range(q), order):
            L = list(L)
            S_L = S[:, L]
            unique_vals = [np.unique(S[:, j]) for j in L]
            
            for a in product(*unique_vals):
                mask_a = np.all(S_L == np.array(a), axis=1)
                n_L_a = mask_a.sum()
                
                if n_L_a == 0:
                    continue
                weight = n_L_a / n
                
                p_L_a_soft = soft_assignments[mask_a].mean(axis=0)  # (K,)
                
                for k in range(K):
                    delta = abs(p_L_a_soft[k] - p_hat_soft[k])
                    
                    mp_deltas_soft.append(delta)
                    mp_max_soft = max(mp_max_soft, delta)
                    mp_wsum_soft += weight * delta
                    mp_wmax_soft = max(mp_wmax_soft, weight * delta)
        
        num_terms_soft = len(mp_deltas_soft) if mp_deltas_soft else 1
        mp_sum_soft = sum(mp_deltas_soft) if mp_deltas_soft else 0.0
        mp_avg_soft = mp_sum_soft / num_terms_soft if mp_deltas_soft else 0.0
        
        results[f'MP{order}_sum(soft)'] = float(mp_sum_soft)
        results[f'MP{order}_max(soft)'] = float(mp_max_soft)
        results[f'MP{order}_avg(soft)'] = float(mp_avg_soft)
        results[f'MP{order}_wsum(soft)'] = float(mp_wsum_soft)
        results[f'MP{order}_wmax(soft)'] = float(mp_wmax_soft)
    
    return results


def compute_SP(S: np.ndarray, cluster_ids: np.ndarray, K: int,
               soft_assignments: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute SP - Subgroup Parity (both hard and soft versions).
    
    Hard version: uses cluster_ids (hard assignment)
    Soft version: uses soft_assignments (soft assignment probabilities)
    
    Definitions:
        p_hat_k = (1/n) * sum_i I(A(x_i)_k = 1)  [global cluster proportion]
        p_hat_k_s = (1/n_s) * sum_{i:s_i=s} I(A(x_i)_k = 1)  [subgroup cluster proportion]
        Delta_k_s = |p_hat_k - p_hat_k_s|
        
        SP^sum = sum_s sum_k Delta_k_s
        SP^max = max_s,k Delta_k_s
        SP^wsum = sum_s sum_k (n_s/n) * Delta_k_s
        SP^wmax = max_s max_k (n_s/n) * Delta_k_s
    """
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    
    n = S.shape[0]
    subgroups = np.array([''.join(map(str, row)) for row in S])
    unique_subgroups = np.unique(subgroups)
    
    results = {}
    
    # --- Hard SP (using cluster_ids) ---
    p_hat_hard = np.zeros(K)
    for k in range(K):
        p_hat_hard[k] = (cluster_ids == k).sum() / n
    
    sp_deltas_hard = []
    sp_max_hard = 0.0
    sp_wsum_hard = 0.0
    sp_wmax_hard = 0.0
    
    for s in unique_subgroups:
        mask_s = subgroups == s
        n_s = mask_s.sum()
        if n_s == 0:
            continue
        weight = n_s / n
        
        for k in range(K):
            n_s_k = ((cluster_ids == k) & mask_s).sum()
            p_hat_k_s = n_s_k / n_s
            delta = abs(p_hat_hard[k] - p_hat_k_s)
            
            sp_deltas_hard.append(delta)
            sp_max_hard = max(sp_max_hard, delta)
            sp_wsum_hard += weight * delta
            sp_wmax_hard = max(sp_wmax_hard, weight * delta)
    
    num_terms_hard = len(sp_deltas_hard) if sp_deltas_hard else 1
    sp_sum_hard = sum(sp_deltas_hard) if sp_deltas_hard else 0.0
    sp_avg_hard = sp_sum_hard / num_terms_hard if sp_deltas_hard else 0.0
    
    results['SP_sum(hard)'] = float(sp_sum_hard)
    results['SP_max(hard)'] = float(sp_max_hard)
    results['SP_avg(hard)'] = float(sp_avg_hard)
    results['SP_wsum(hard)'] = float(sp_wsum_hard)
    results['SP_wmax(hard)'] = float(sp_wmax_hard)
    
    # --- Soft SP (using soft_assignments) ---
    if soft_assignments is not None:
        p_hat_soft = soft_assignments.mean(axis=0)  # (K,)
        
        sp_deltas_soft = []
        sp_max_soft = 0.0
        sp_wsum_soft = 0.0
        sp_wmax_soft = 0.0
        
        for s in unique_subgroups:
            mask_s = subgroups == s
            n_s = mask_s.sum()
            if n_s == 0:
                continue
            weight = n_s / n
            
            p_hat_k_s = soft_assignments[mask_s].mean(axis=0)  # (K,)
            
            for k in range(K):
                delta = abs(p_hat_soft[k] - p_hat_k_s[k])
                
                sp_deltas_soft.append(delta)
                sp_max_soft = max(sp_max_soft, delta)
                sp_wsum_soft += weight * delta
                sp_wmax_soft = max(sp_wmax_soft, weight * delta)
        
        num_terms_soft = len(sp_deltas_soft) if sp_deltas_soft else 1
        sp_sum_soft = sum(sp_deltas_soft) if sp_deltas_soft else 0.0
        sp_avg_soft = sp_sum_soft / num_terms_soft if sp_deltas_soft else 0.0
        
        results['SP_sum(soft)'] = float(sp_sum_soft)
        results['SP_max(soft)'] = float(sp_max_soft)
        results['SP_avg(soft)'] = float(sp_avg_soft)
        results['SP_wsum(soft)'] = float(sp_wsum_soft)
        results['SP_wmax(soft)'] = float(sp_wmax_soft)
    
    return results


def compute_WMP(S: np.ndarray, soft_assignments: np.ndarray, K: int) -> Dict[str, float]:
    """
    Compute WMP - Wasserstein Marginal Parity (soft clustering only).
    
    For each cluster k, we compare the distribution of A(x_i)_k between 
    the full population and each marginal subgroup.
    
    W_{j,a,k} = W_1({A_{ik} : S_{ij}=a}, {A_{ik} : all})
    
    Aggregations (following Table 2 conventions):
        WMP_max  = max_{j,a,k} W_{j,a,k}                    (main, unweighted)
        WMP_wmax = max_{j,a,k} (n_{j,a}/n) * W_{j,a,k}      (main, weighted)
        WMP_wsum = sum_{j,a,k} (n_{j,a}/n) * W_{j,a,k}      (average deviation)
        WMP_avg  = (1 / num_terms) * sum_{j,a,k} W_{j,a,k}  (normalized sum)
    
    Note: WMP_sum is K-dependent and not directly comparable across different K.
          Use WMP_avg for normalized comparison.
    """
    if soft_assignments is None:
        return {}
    
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    
    n, q = S.shape
    
    wmp_values = []  # Store all W_{j,a,k} for averaging
    wmp_max = 0.0
    wmp_wsum = 0.0
    wmp_wmax = 0.0
    
    for j in range(q):
        attr_j = S[:, j]
        unique_vals = np.unique(attr_j)
        
        for k in range(K):
            A_k_all = soft_assignments[:, k]
            
            for a in unique_vals:
                mask_a = attr_j == a
                n_a = mask_a.sum()
                
                if n_a == 0:
                    continue
                weight = n_a / n
                
                A_k_a = soft_assignments[mask_a, k]
                w1 = wasserstein_distance(A_k_a, A_k_all)
                
                wmp_values.append(w1)
                wmp_max = max(wmp_max, w1)
                wmp_wsum += weight * w1
                wmp_wmax = max(wmp_wmax, weight * w1)
    
    num_terms = len(wmp_values) if wmp_values else 1
    wmp_sum = sum(wmp_values) if wmp_values else 0.0
    wmp_avg = wmp_sum / num_terms if wmp_values else 0.0
    
    return {
        'WMP_sum': float(wmp_sum),      # Raw sum (K-dependent)
        'WMP_max': float(wmp_max),      # Main: unweighted max
        'WMP_avg': float(wmp_avg),      # Normalized: sum / num_terms
        'WMP_wsum': float(wmp_wsum),    # Optional: weighted sum
        'WMP_wmax': float(wmp_wmax),    # Main: weighted max (Table 2 style)
    }


def compute_all(X: np.ndarray, y: np.ndarray, S: np.ndarray, 
                cluster_ids: np.ndarray, K: int,
                soft_assignments: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute all metrics (hard and soft versions)."""
    marginal = compute_marginal_balance(S, cluster_ids, K, soft_assignments=soft_assignments)
    subgroup = compute_subgroup_balance(S, cluster_ids, K, soft_assignments=soft_assignments)
    
    metrics = {
        'Cost': round(compute_cost(X, cluster_ids, K), 4),
        'Acc': round(compute_acc(y, cluster_ids), 4),
        'ARI': round(sklearn_metrics.adjusted_rand_score(y, cluster_ids), 4),
        'NMI': round(sklearn_metrics.normalized_mutual_info_score(y, cluster_ids), 4),
        'Delta': round(compute_delta(S, cluster_ids, K), 4),
        'MarginalBalance': round(marginal['min'], 4),
        'SubgroupBalance': round(subgroup['min_balance'], 4),
        # Average subgroup balance across clusters (less brittle than min over clusters)
        'SubgroupBalance_avg': round(subgroup.get('avg_balance', 0.0), 4),
    }
    
    # Add soft balance metrics if available
    if soft_assignments is not None:
        metrics['MarginalBalance(soft)'] = round(marginal.get('min(soft)', 0.0), 4)
        metrics['SubgroupBalance(soft)'] = round(subgroup.get('min_balance(soft)', 0.0), 4)
        metrics['SubgroupBalance_avg(soft)'] = round(subgroup.get('avg_balance(soft)', 0.0), 4)
    
    # Gap metrics (hard and soft) - Fairness Gap between groups
    gap = compute_gap(S, cluster_ids, K, soft_assignments=soft_assignments)
    for key, val in gap.items():
        metrics[key] = round(val, 4)
    
    # MP metrics (hard and soft)
    mp1 = compute_MP(S, cluster_ids, K, order=1, soft_assignments=soft_assignments)
    mp2 = compute_MP(S, cluster_ids, K, order=2, soft_assignments=soft_assignments)
    for key, val in mp1.items():
        metrics[key] = round(val, 4)
    for key, val in mp2.items():
        metrics[key] = round(val, 4)
    
    # SP metrics (hard and soft)
    sp = compute_SP(S, cluster_ids, K, soft_assignments=soft_assignments)
    for key, val in sp.items():
        metrics[key] = round(val, 4)
    
    # WMP metrics (soft only)
    if soft_assignments is not None:
        wmp = compute_WMP(S, soft_assignments, K)
        for key, val in wmp.items():
            metrics[key] = round(val, 4)
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Results") -> None:
    """Print metrics in formatted table."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    print(f"  Cost:            {metrics.get('Cost', 'N/A')}")
    print(f"  Balance:         {metrics.get('SubgroupBalance', 'N/A')}")
    print(f"  MarginalBalance: {metrics.get('MarginalBalance', 'N/A')}")
    print(f"  Delta:           {metrics.get('Delta', 'N/A')}")
    if 'SubgroupBalance(soft)' in metrics:
        print(f"  Balance(soft):         {metrics.get('SubgroupBalance(soft)', 'N/A')}")
        print(f"  MarginalBalance(soft): {metrics.get('MarginalBalance(soft)', 'N/A')}")
    print()
    print("  --- Fairness Gap (Δ) ---")
    print(f"  Gap_max(hard):   {metrics.get('Gap_max(hard)', 'N/A')}")
    print(f"  Gap_wmax(hard):  {metrics.get('Gap_wmax(hard)', 'N/A')}")
    print(f"  Gap_avg(hard):   {metrics.get('Gap_avg(hard)', 'N/A')}")
    if 'Gap_max(soft)' in metrics:
        print(f"  Gap_max(soft):   {metrics.get('Gap_max(soft)', 'N/A')}")
        print(f"  Gap_wmax(soft):  {metrics.get('Gap_wmax(soft)', 'N/A')}")
        print(f"  Gap_avg(soft):   {metrics.get('Gap_avg(soft)', 'N/A')}")
    print()
    print("  --- Hard Fairness Metrics (Table 2 style) ---")
    print(f"  MP1_max(hard):   {metrics.get('MP1_max(hard)', 'N/A')}")
    print(f"  MP1_wmax(hard):  {metrics.get('MP1_wmax(hard)', 'N/A')}")
    print(f"  MP2_max(hard):   {metrics.get('MP2_max(hard)', 'N/A')}")
    print(f"  MP2_wmax(hard):  {metrics.get('MP2_wmax(hard)', 'N/A')}")
    print(f"  SP_max(hard):    {metrics.get('SP_max(hard)', 'N/A')}")
    print(f"  SP_wmax(hard):   {metrics.get('SP_wmax(hard)', 'N/A')}")
    print()
    print("  --- Soft Fairness Metrics ---")
    if 'MP1_max(soft)' in metrics:
        print(f"  MP1_max(soft):   {metrics.get('MP1_max(soft)', 'N/A')}")
        print(f"  MP1_wmax(soft):  {metrics.get('MP1_wmax(soft)', 'N/A')}")
        print(f"  MP2_max(soft):   {metrics.get('MP2_max(soft)', 'N/A')}")
        print(f"  MP2_wmax(soft):  {metrics.get('MP2_wmax(soft)', 'N/A')}")
        print(f"  SP_max(soft):    {metrics.get('SP_max(soft)', 'N/A')}")
        print(f"  SP_wmax(soft):   {metrics.get('SP_wmax(soft)', 'N/A')}")
        print(f"  WMP_max:         {metrics.get('WMP_max', 'N/A')}")
        print(f"  WMP_wmax:        {metrics.get('WMP_wmax', 'N/A')}")
    print()
    print(f"  Acc:             {metrics.get('Acc', 'N/A')}")
    print(f"  ARI:             {metrics.get('ARI', 'N/A')}")
    print(f"  NMI:             {metrics.get('NMI', 'N/A')}")
    print(f"{'='*70}\n")
