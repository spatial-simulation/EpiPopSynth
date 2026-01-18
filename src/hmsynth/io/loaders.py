"""Survey loading and standardization."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from hmsynth.io.schema import SchemaConfig


def _build_gender_lookup(mapping_cfg) -> dict:
    lookup = {}
    for k, vals in mapping_cfg.items():
        for v in vals:
            lookup[v] = 0 if k.lower() == "female" else 1
    return lookup


def _map_gender(series: pd.Series, lookup: dict) -> np.ndarray:
    mapped = []
    for v in series.to_numpy():
        if v in lookup:
            mapped.append(lookup[v])
        elif isinstance(v, str) and v.lower() in lookup:
            mapped.append(lookup[v.lower()])
        else:
            raise ValueError(f"Unknown gender value: {v}")
    return np.asarray(mapped, dtype=int)


def _to_age_bin(age_values: np.ndarray, age_type: str, bins) -> np.ndarray:
    if age_type == "continuous":
        return np.searchsorted(np.asarray(bins, dtype=int), age_values, side="right")
    elif age_type == "binned":
        return age_values.astype(int)
    else:
        raise ValueError(f"Unsupported age type: {age_type}")


def load_survey(schema: SchemaConfig, path: Optional[str] = None) -> pd.DataFrame:
    """Load survey CSV and standardize to columns: household_id, sex, age_bin (and pid if present)."""
    csv_path = path or schema.survey_path
    df = pd.read_csv(csv_path)

    required = [schema.field_household_id, schema.field_gender, schema.field_age]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column in survey: {col}")

    # Rename to standard names
    rename_map = {
        schema.field_household_id: "household_id",
        schema.field_gender: "gender_raw",
        schema.field_age: "age_raw",
    }
    if schema.field_person_id and schema.field_person_id in df.columns:
        rename_map[schema.field_person_id] = "pid"

    df = df.rename(columns=rename_map)

    # Gender mapping
    gender_lookup = _build_gender_lookup(schema.gender_mapping)
    df["sex"] = _map_gender(df["gender_raw"], gender_lookup)

    # Age binning
    age_vals = pd.to_numeric(df["age_raw"], errors="coerce").to_numpy()
    if np.isnan(age_vals).any():
        raise ValueError("Non-numeric age encountered after coercion")
    df["age_bin"] = _to_age_bin(age_vals, schema.age.type, schema.age.bins)

    # Keep standard columns
    keep_cols = ["household_id", "sex", "age_bin"]
    if "pid" in df.columns:
        keep_cols.insert(0, "pid")
    return df[keep_cols]

