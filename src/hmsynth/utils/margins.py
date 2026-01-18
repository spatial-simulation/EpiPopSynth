"""Compute marginal distributions from survey and motifs."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_margins(survey_df: pd.DataFrame, motif_df: pd.DataFrame, max_hsize: int) -> pd.DataFrame:
    """
    计算 gender/age_bin/household_size 的目标边际分布。
    - gender, age_bin 来自标准化个体表
    - household_size 使用 motif 分布（h_size_cap * freq）
    """
    # gender margin
    gender_counts = survey_df["sex"].value_counts().reindex([0, 1], fill_value=0).to_numpy()
    n_female, n_male = gender_counts

    # age margin
    n_age_bins = survey_df["age_bin"].max() + 1
    age_counts = np.bincount(survey_df["age_bin"], minlength=n_age_bins)

    # household size margin
    h_sizes = motif_df["h_size_cap"].to_numpy()
    freqs = motif_df["freq"].to_numpy()
    hsize_counts = np.zeros(max_hsize, dtype=int)
    for hs, f in zip(h_sizes, freqs):
        idx = min(hs, max_hsize) - 1
        hsize_counts[idx] += f

    # assemble DataFrame with single row
    col_g = ["n_female", "n_male"]
    col_a = [f"n_a{a}" for a in range(n_age_bins)]
    col_s = [f"n_s{s}" for s in range(max_hsize)]
    data = np.r_[gender_counts, age_counts, hsize_counts].reshape(1, -1)
    cols = col_g + col_a + col_s
    return pd.DataFrame(data, columns=cols)

