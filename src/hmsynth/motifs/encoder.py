"""Encode raw motifs into feature-rich structures."""
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from hmsynth.models.motif import EncodedMotif, RawMotif


@profile
def encode_motifs(
    motifs: Iterable[RawMotif],
    age_bins: Sequence[int],
    max_hsize: int,
) -> List[EncodedMotif]:
    age_bins = np.asarray(age_bins, dtype=int)
    n_age = len(age_bins) + 1
    max_hsize = int(max_hsize)

    encoded: List[EncodedMotif] = []
    for m in motifs:
        arr_age = np.zeros(n_age, dtype=int)
        arr_gender = np.zeros(2, dtype=int)
        h_raw = 0
        for sex, a_bin, n in m.code:
            arr_gender[sex] += n
            arr_age[a_bin] += n
            h_raw += n
        h_cap = min(h_raw, max_hsize)
        h_onehot = np.zeros(max_hsize, dtype=int)
        h_onehot[h_cap - 1] = 1
        encoded.append(
            EncodedMotif(
                code=m.code,
                freq=m.freq,
                h_size_raw=h_raw,
                h_size_cap=h_cap,
                n_female=int(arr_gender[0]),
                n_male=int(arr_gender[1]),
                age_counts=arr_age,
                hsize_onehot=h_onehot,
            )
        )
    return encoded

