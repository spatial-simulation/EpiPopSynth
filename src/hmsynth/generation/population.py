"""High-level population generation helpers."""
from __future__ import annotations

import os
import pandas as pd

from hmsynth.generation.sampler import sample_households, households_to_persons


def generate_population(motif_df: pd.DataFrame, best_x, population_size: int, seed: int = 42) -> pd.DataFrame:
    sampled = sample_households(motif_df, best_x, population_size, seed=seed)
    persons = households_to_persons(sampled, seed=seed)
    return persons


def save_population(df: pd.DataFrame, path: str):
    dirpath = os.path.dirname(path) or "."
    if os.path.isfile(dirpath):
        raise ValueError(f"Cannot create directory '{dirpath}' because a file with the same name exists.")
    os.makedirs(dirpath, exist_ok=True)
    df.to_csv(path, index=False)

