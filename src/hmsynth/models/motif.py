"""Data models for household motifs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

MotifCode = Tuple[Tuple[int, int, int], ...]  # ((sex, age_bin, count), ...)


@dataclass(frozen=True)
class RawMotif:
    code: MotifCode
    freq: int  # motif 出现的户数


@dataclass(frozen=True)
class EncodedMotif:
    code: MotifCode
    freq: int
    h_size_raw: int           # 未截断户规模
    h_size_cap: int           # 截断到 max_hsize
    n_female: int
    n_male: int
    age_counts: np.ndarray    # shape = (n_age_bins,)
    hsize_onehot: np.ndarray  # shape = (max_hsize,)

