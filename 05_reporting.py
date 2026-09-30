"""
05_reporting.py
===============
Posterior difference between the pooled exposure slopes of the G2 (C1) and
G1 (C2) analyses for every regression estimand, with 95% HDI (manuscript
Table 5 and Supplementary Table S4).

Output: Results/Table_C1_vs_C2_gap.csv
"""

import csv
import logging
from pathlib import Path

import numpy as np
import arviz as az
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "Results"
SPEC_FILE   = BASE_DIR / "config" / "model_specs.yaml"


def hdi_95(draws):
    lo, hi = az.hdi(np.asarray(draws, dtype=float), hdi_prob=0.95)
    return float(lo), float(hi)


def load_beta_exp(outcome, variant, model_id):
    idata = az.from_netcdf(str(RESULTS_DIR / f"{outcome}_{variant}" / f"{model_id}_idata.nc"))
    return idata.posterior["beta_exp"].values.reshape(-1)


def main():
    specs   = yaml.safe_load(open(SPEC_FILE))
    records = []
    for outcome in ("FA", "MD"):
        for model_id, spec in specs[outcome].items():
            if spec["type"] != "regression":
                continue
            s1   = load_beta_exp(outcome, "C1", model_id)
            s2   = load_beta_exp(outcome, "C2", model_id)
            diff = s1 - s2
            lo, hi = hdi_95(diff)
            records.append({
                "outcome":       outcome,
                "model":         model_id,
                "mean_C1":       float(s1.mean()),
                "mean_C2":       float(s2.mean()),
                "diff":          float(diff.mean()),
                "hdi_diff_2.5":  lo,
                "hdi_diff_97.5": hi,
            })

    out_path = RESULTS_DIR / "Table_C1_vs_C2_gap.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    log.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
