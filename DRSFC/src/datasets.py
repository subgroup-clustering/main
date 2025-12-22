"""
Dataset loading utilities for DRSFC.

Supports preprocessed .npy files with structure:
    data/{dataset_name}/
        X.npy       # Features (n, d)
        y.npy       # Labels (n,)
        S.npy       # Sensitive attributes (n, q)
"""

import numpy as np
import os
from pathlib import Path
from typing import Dict, Any, Optional, List


def L2_normalize(X: np.ndarray) -> np.ndarray:
    """L2-normalize each row of X, with stability threshold."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-7)
    return X / norms


def load_data(
    name: str,
    l2_normalize: bool = False,
    data_dir: Optional[str] = None
) -> tuple:
    """
    Load preprocessed dataset from .npy files.
    
    Args:
        name: Dataset name (e.g., 'adult(2)', 'dutch(2)', 'communities(18)')
        l2_normalize: Whether to L2-normalize features
        data_dir: Optional path to data directory
    
    Returns:
        Tuple of (X, S, y, K, d, q)
            - X: Features (n, d)
            - S: Sensitive attributes (n, q)
            - y: Labels (n,)
            - K: Suggested number of clusters (from unique labels)
            - d: Feature dimension
            - q: Number of sensitive attributes
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / 'data'
    else:
        data_dir = Path(data_dir)
    
    folder = data_dir / name
    
    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")
    
    # Load data
    X = np.load(folder / 'X.npy').astype(np.float32)
    y = np.load(folder / 'y.npy')
    S = np.load(folder / 'S.npy')
    
    # Normalize if requested
    if l2_normalize:
        X = L2_normalize(X)
    
    n, d = X.shape
    q = S.shape[1] if S.ndim > 1 else 1
    K = len(np.unique(y))
    
    print(f"[Data] Loaded '{name}': n={n}, d={d}, q={q}, K_suggested={K}")
    
    return X, S, y, K, d, q


def get_subgroups(S: np.ndarray) -> np.ndarray:
    """
    Convert multi-attribute sensitive matrix to subgroup identifiers.
    
    Args:
        S: Sensitive attributes (n, q)
    
    Returns:
        Subgroup identifiers (n,) as integers
    """
    if S.ndim == 1:
        return S
    
    # Convert each row to unique integer
    n, q = S.shape
    unique_rows = {}
    subgroups = np.zeros(n, dtype=np.int32)
    
    for i in range(n):
        key = tuple(S[i])
        if key not in unique_rows:
            unique_rows[key] = len(unique_rows)
        subgroups[i] = unique_rows[key]
    
    return subgroups


def split_by_color(X: np.ndarray, S: np.ndarray) -> List[np.ndarray]:
    """
    Split data by sensitive attribute (color) for FCA-style processing.
    
    Args:
        X: Features (n, d)
        S: Sensitive attributes (n,) or (n, q)
    
    Returns:
        List of X arrays, one per unique sensitive value/subgroup
    """
    subgroups = get_subgroups(S)
    n_color = len(np.unique(subgroups))
    
    xs = [X[subgroups == c] for c in range(n_color)]
    
    for idx, arr in enumerate(xs):
        print(f"[Data] Color {idx} shape: {arr.shape}")
    
    return xs
