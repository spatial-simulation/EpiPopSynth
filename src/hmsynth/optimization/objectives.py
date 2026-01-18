"""Objective construction for motif weight optimization."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def build_matrices(motif_df) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从 motif DataFrame 构造 W_a, W_g, W_s."""
    cols_a = [c for c in motif_df.columns if c.startswith("n_a")]
    cols_s = [c for c in motif_df.columns if c.startswith("n_s")]
    W_a = motif_df[cols_a].to_numpy(dtype=float)
    W_s = motif_df[cols_s].to_numpy(dtype=float)
    W_g = motif_df[["n_female", "n_male"]].to_numpy(dtype=float)
    return W_a, W_g, W_s


def build_targets(margin_df) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从边际 DataFrame 构造 y_a, y_g, y_s."""
    cols_a = [c for c in margin_df.columns if c.startswith("n_a")]
    cols_s = [c for c in margin_df.columns if c.startswith("n_s")]
    y_a = margin_df[cols_a].to_numpy(dtype=float).ravel()
    y_s = margin_df[cols_s].to_numpy(dtype=float).ravel()
    y_g = margin_df[["n_female", "n_male"]].to_numpy(dtype=float).ravel()
    return y_a, y_g, y_s


def residuals(x, W_a, W_g, W_s, y_a, y_g, y_s, x_init, reg_strength: float):
    """组合残差向量，供 least_squares 使用。"""
    res_a = y_a - W_a.T @ x
    res_g = y_g - W_g.T @ x
    res_s = y_s - W_s.T @ x
    res_m = (x - x_init) * reg_strength
    return np.concatenate([res_a, res_g, res_s, res_m])

