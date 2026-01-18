"""Regularized non-negative least squares for motif weights."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import least_squares, lsq_linear
from scipy import sparse

from hmsynth.optimization.objectives import build_matrices, build_targets, residuals


@dataclass
class OptimConfig:
    reg_strength: float
    max_iter: int
    ftol: float
    xtol: float
    gtol: float
    bounds_scale: float
    seed: Optional[int]
    lognormal_noise_std: float


@profile
def solve_weights(motif_df, margin_df, cfg: OptimConfig):
    W_a, W_g, W_s = build_matrices(motif_df)
    y_a, y_g, y_s = build_targets(margin_df)

    # 初值：x_init（户数，非比例）
    x_init_prop = motif_df["x_init"].to_numpy(dtype=float)
    hsize_cap = motif_df["h_size_cap"].to_numpy(dtype=float)
    n_total = y_g.sum()  # 总人口数
    x_init = np.round(x_init_prop * (n_total / (x_init_prop @ hsize_cap))).astype(float)

    rng = np.random.default_rng(cfg.seed)
    if cfg.lognormal_noise_std > 0:
        x0 = x_init * rng.lognormal(mean=0.0, sigma=cfg.lognormal_noise_std, size=x_init.shape)
    else:
        x0 = x_init.copy()

    lower = np.zeros_like(x0)
    upper = np.full_like(x0, x_init.max() * cfg.bounds_scale)

    # 增广矩阵一次求解（线性问题）
    # 转为稀疏矩阵，加速求解
    A_blocks = [sparse.csr_matrix(W_a.T), sparse.csr_matrix(W_g.T), sparse.csr_matrix(W_s.T)]
    b_blocks = [y_a, y_g, y_s]
    if cfg.reg_strength > 0:
        lam_sqrt = np.sqrt(cfg.reg_strength)
        A_blocks.append(lam_sqrt * sparse.eye(len(x0), format="csr"))
        b_blocks.append(lam_sqrt * x_init)

    A = sparse.vstack(A_blocks, format="csr")
    b = np.concatenate(b_blocks)

    res = lsq_linear(
        A,
        b,
        bounds=(lower, upper),
        max_iter=cfg.max_iter,
        tol=cfg.ftol,
        verbose=0,
    )

    best_x = np.round(np.clip(res.x, 0, None)).astype(int)
    return best_x, res

