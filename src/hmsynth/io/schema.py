"""Schema config loading and validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import yaml


@dataclass
class AgeConfig:
    type: str  # "continuous" or "binned"
    bins: Sequence[int]


@dataclass
class SchemaConfig:
    survey_path: str
    field_person_id: Optional[str]
    field_household_id: str
    field_gender: str
    field_age: str
    gender_mapping: Dict[str, List]
    age: AgeConfig
    max_household_size: int


def load_schema(path: str) -> SchemaConfig:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    input_cfg = cfg.get("input", {})
    fields_cfg = cfg.get("fields", {})
    gender_mapping = cfg.get("gender_mapping", {})
    age_cfg = cfg.get("age", {})

    survey_path = input_cfg.get("survey_path")
    if not survey_path:
        raise ValueError("survey_path is required in configs/schema.yaml")

    field_pid = fields_cfg.get("person_id")
    field_hid = fields_cfg.get("household_id")
    field_gender = fields_cfg.get("gender")
    field_age = fields_cfg.get("age")
    if not field_hid or not field_gender or not field_age:
        raise ValueError("fields.household_id, fields.gender, fields.age are required")

    age_type = age_cfg.get("type", "continuous")
    bins = age_cfg.get("bins", [])
    max_hsize = int(cfg.get("max_household_size", 6))

    return SchemaConfig(
        survey_path=survey_path,
        field_person_id=field_pid,
        field_household_id=field_hid,
        field_gender=field_gender,
        field_age=field_age,
        gender_mapping=gender_mapping,
        age=AgeConfig(type=age_type, bins=bins),
        max_household_size=max_hsize,
    )

