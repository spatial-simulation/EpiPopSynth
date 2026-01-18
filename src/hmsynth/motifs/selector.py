"""Select representative motifs and attach initial weights."""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

from hmsynth.models.motif import EncodedMotif


@profile
def select_motifs(
    motifs: Iterable[EncodedMotif],
    coverage_threshold: float = 0.99,
    ensure_person_type: bool = True,
    ensure_hsize: bool = True,
) -> pd.DataFrame:
    """返回筛选后的 DataFrame，含 x_init 列。"""
    motifs = list(motifs)
    if len(motifs) == 0:
        return pd.DataFrame()

    counts = np.array([m.freq for m in motifs], dtype=float)
    cdf = np.cumsum(counts / counts.sum())
    n_prop = np.searchsorted(cdf, coverage_threshold)

    n_ptype = 0
    if ensure_person_type:
        needed = set((g, a) for m in motifs for g, a, n in m.code)
        seen = set()
        idx = 0
        for i, m in enumerate(motifs):
            seen.update((g, a) for g, a, n in m.code)
            if needed.issubset(seen):
                idx = i
                break
        n_ptype = idx

    n_hsize = 0
    if ensure_hsize:
        needed_h = set(m.h_size_raw for m in motifs)
        seen_h = set()
        idx = 0
        for i, m in enumerate(motifs):
            seen_h.add(m.h_size_raw)
            if needed_h.issubset(seen_h):
                idx = i
                break
        n_hsize = idx

    n_keep = max(n_prop, n_ptype, n_hsize)
    motifs_keep = motifs[: n_keep + 1]

    total = sum(m.freq for m in motifs_keep)
    rows = []
    # 推断 age_bins / max_hsize 维度
    n_age = len(motifs_keep[0].age_counts)
    max_hsize = len(motifs_keep[0].hsize_onehot)
    col_a = [f"n_a{a}" for a in range(n_age)]
    col_s = [f"n_s{s}" for s in range(max_hsize)]

    for m in motifs_keep:
        rows.append(
            dict(
                h_code=str(tuple((int(g), int(a), int(n)) for g, a, n in m.code)),
                freq=int(m.freq),
                x_init=float(m.freq / total),
                h_size_raw=int(m.h_size_raw),
                h_size_cap=int(m.h_size_cap),
                n_female=int(m.n_female),
                n_male=int(m.n_male),
                **{k: int(v) for k, v in zip(col_a, m.age_counts)},
                **{k: int(v) for k, v in zip(col_s, m.hsize_onehot)},
            )
        )

    return pd.DataFrame(rows)

