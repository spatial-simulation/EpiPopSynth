"""Extract household motifs from standardized survey data."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, List

import numpy as np
import pandas as pd

from hmsynth.models.motif import MotifCode, RawMotif


@profile
def extract_raw_motifs(df: pd.DataFrame) -> List[RawMotif]:
    """
    输入: DataFrame，需包含 household_id, sex, age_bin (已标准化/分箱)
    输出: RawMotif 列表，按频次降序
    """
    if not {"household_id", "sex", "age_bin"}.issubset(df.columns):
        raise ValueError("DataFrame must contain household_id, sex, age_bin")

    mat = np.c_[
        df["household_id"].to_numpy(),
        df["sex"].to_numpy(),
        df["age_bin"].to_numpy(),
    ]
    mat = mat[mat[:, 0].argsort()]
    _, idx = np.unique(mat[:, 0], return_index=True)
    households = np.split(mat, idx[1:])

    motif_counts: Counter[MotifCode] = Counter()
    for hh in households:
        c = Counter(map(tuple, hh[:, (1, 2)]))  # (sex, age_bin) -> count
        code: MotifCode = tuple(sorted((g, a, n) for (g, a), n in c.items()))
        motif_counts[code] += 1

    # 排序按频次降序
    sorted_items = sorted(motif_counts.items(), key=lambda kv: kv[1], reverse=True)
    return [RawMotif(code=code, freq=freq) for code, freq in sorted_items]

