"""Output helpers."""
from __future__ import annotations

import os
import pandas as pd


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_dataframe(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path) or ".")
    df.to_csv(path, index=False)

