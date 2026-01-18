"""End-to-end pipeline to synthesize population via household motifs."""
from __future__ import annotations

import os
import yaml
import numpy as np
import pandas as pd

from hmsynth.io.schema import load_schema
from hmsynth.io.loaders import load_survey
from hmsynth.io.writers import write_dataframe
from hmsynth.motifs.extractor import extract_raw_motifs
from hmsynth.motifs.encoder import encode_motifs
from hmsynth.motifs.selector import select_motifs
from hmsynth.optimization.optimizer import OptimConfig, solve_weights
from hmsynth.utils.margins import compute_margins
from hmsynth.generation.population import generate_population, save_population


def _load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run_pipeline(
    config_dir: str = "configs",
    survey_path: str | None = None,
    output_dir: str = "outputs",
):
    # 1) Load configs
    schema_cfg = load_schema(os.path.join(config_dir, "schema.yaml"))
    motif_cfg = _load_yaml(os.path.join(config_dir, "motif.yaml"))
    opt_cfg_raw = _load_yaml(os.path.join(config_dir, "optimization.yaml"))
    gen_cfg = _load_yaml(os.path.join(config_dir, "generation.yaml"))

    # 2) Load survey
    survey_df = load_survey(schema_cfg, path=survey_path)

    # 3) Build motifs
    raw_motifs = extract_raw_motifs(survey_df)
    enc_motifs = encode_motifs(raw_motifs, age_bins=schema_cfg.age.bins, max_hsize=schema_cfg.max_household_size)
    motif_df = select_motifs(
        enc_motifs,
        coverage_threshold=motif_cfg["selection"]["coverage_threshold"],
        ensure_person_type=motif_cfg["selection"]["ensure_person_type_coverage"],
        ensure_hsize=motif_cfg["selection"]["ensure_household_size_coverage"],
    )

    # 4) Compute margins
    margin_df = compute_margins(survey_df, motif_df, max_hsize=schema_cfg.max_household_size)

    # 5) Optimize weights
    opt_cfg = OptimConfig(
        reg_strength=float(opt_cfg_raw["regularization"]["strength"]),
        max_iter=int(opt_cfg_raw["solver"]["max_iter"]),
        ftol=float(opt_cfg_raw["solver"]["ftol"]),
        xtol=float(opt_cfg_raw["solver"]["xtol"]),
        gtol=float(opt_cfg_raw["solver"]["gtol"]),
        bounds_scale=float(opt_cfg_raw["bounds"]["scale"]),
        seed=opt_cfg_raw["random"]["seed"],
        lognormal_noise_std=float(opt_cfg_raw["random"]["lognormal_noise_std"]),
    )
    best_x, _ = solve_weights(motif_df, margin_df, opt_cfg)

    # 6) Generate population
    pop_size = gen_cfg["population_size"]
    pop_seed = gen_cfg.get("random_seed", 42)
    pop_df = generate_population(motif_df, best_x, pop_size, seed=pop_seed)

    # 7) Save outputs
    os.makedirs(output_dir, exist_ok=True)
    motif_out = os.path.join(output_dir, "motifs.csv")
    margin_out = os.path.join(output_dir, "margins.csv")
    bestx_out = os.path.join(output_dir, "best_x.npy")
    pop_out = os.path.join(gen_cfg["output"]["path"])
    write_dataframe(motif_df, motif_out)
    write_dataframe(margin_df, margin_out)
    np.save(bestx_out, best_x)
    save_population(pop_df, pop_out)

    return {
        "motifs": motif_out,
        "margins": margin_out,
        "best_x": bestx_out,
        "population": pop_out,
    }

