"""Sample motifs into household/person microdata."""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


@profile
def sample_households(motif_df: pd.DataFrame, best_x: np.ndarray, population_size: int, seed: int = 42):
    rng = np.random.default_rng(seed)

    p = best_x / best_x.sum()
    hsize_cap = motif_df["h_size_cap"].to_numpy()
    avg_hsize = float((hsize_cap * p).sum())
    N = int(round(population_size / avg_hsize))

    motifs = motif_df["h_code"].to_numpy()
    sampled = rng.choice(motifs, size=N, p=p)
    return sampled


@profile
def households_to_persons(sampled_motifs: Iterable, seed: int = 42) -> pd.DataFrame:
    """展开 motif 为个体表: pid, hid, sex, age_bin"""
    import ast

    records = []
    hid = 0
    pid = 0
    for code in sampled_motifs:
        if isinstance(code, str):
            code = ast.literal_eval(code)
        members = []
        for sex, age_bin, cnt in code:
            members.extend([(sex, age_bin)] * cnt)
        members = np.asarray(members, dtype=int)
        hids = np.full(len(members), hid, dtype=int)
        pids = np.arange(pid, pid + len(members), dtype=int)
        rec = np.c_[pids, hids, members]
        records.append(rec)
        hid += 1
        pid += len(members)

    if not records:
        return pd.DataFrame(columns=["pid", "hid", "sex", "age_bin"])

    mat = np.concatenate(records, axis=0)
    return pd.DataFrame(mat, columns=["pid", "hid", "sex", "age_bin"])

