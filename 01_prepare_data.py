"""
01_prepare_data.py
==================
Build the subpatch-level analysis dataset from the extracted per-block CSVs:
ROI pooling, block-level technical covariates, normalization, patch-size term
and integer indices.

Normalization (global, across all subpatches, blocks and ROI types):
  FA, MD, myelin, cell, axon density, fiber coherence, technical covariates:
      z-score
  GFAP (astrocyte) and microglia density:
      log10(x + eps), then z-score; eps = 5th percentile of the positive values

Block block_64_20_2 is excluded (uncorrectable staining batch effect).

Outputs (Analysis_Dataset/)
---------------------------
analysis_dataset.csv
normalization_params.json
index_maps.json
sample_size_table.csv      subpatch counts, ROI type x block
"""

import csv
import json
import logging
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "Data_Extraction"
CONFIG_DIR = BASE_DIR / "config"
OUT_DIR    = BASE_DIR / "Analysis_Dataset"

MASK_COLUMN = "metric_artifact_mask"

METRICS = (
    "FA",
    "MD",
    "axon_density",
    "axon_coherence",
    "cell_density",
    "gfap_density",
    "microglia_density",
    "myelin_density",
)
TECH_COLS = ("FixDur_days", "StorTime_days")

ZSCORE_COLS     = ("FA", "MD", "myelin_density", "cell_density",
                   "axon_density", "axon_coherence") + TECH_COLS
LOG_ZSCORE_COLS = ("gfap_density", "microglia_density")

EXCLUDED_BLOCKS = {"block_64_20_2"}


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def percentile(values, p):
    """p-th percentile (0-100) with linear interpolation."""
    sv = sorted(values)
    idx = (p / 100) * (len(sv) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sv) - 1)
    return sv[lo] + (sv[hi] - sv[lo]) * (idx - lo)


def mean_std(values):
    n = len(values)
    m = sum(values) / n
    s = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    return m, s


def load_all_csvs(data_dir):
    """Long-format records: block_id, metric, roi_raw, patch_index, value, n_voxels."""
    records = []
    n_col = MASK_COLUMN + "_n_voxels"
    fname_pattern = re.compile(
        r"^block_(?P<block_id>.+?)_(?P<metric>"
        + "|".join(re.escape(m) for m in METRICS)
        + r")\.csv$"
    )

    for fname in sorted(os.listdir(data_dir)):
        m = fname_pattern.match(fname)
        if m is None:
            continue
        block_id = "block_" + m.group("block_id")
        if block_id in EXCLUDED_BLOCKS:
            continue
        metric = m.group("metric")

        with open(data_dir / fname, newline="") as fh:
            for row in csv.DictReader(fh):
                sp = row.get("sub_patch", "").strip()
                if not sp:
                    continue
                # sub_patch = "<raw ROI name>_<patch index>"
                pm = re.match(r"^(.+?)_(\d+)$", sp)
                if pm:
                    roi_raw, patch_index = pm.group(1), int(pm.group(2))
                else:
                    roi_raw, patch_index = sp, 0

                raw_val = row.get(MASK_COLUMN, "").strip()
                raw_n   = row.get(n_col, "").strip()
                try:
                    value = float(raw_val) if raw_val not in ("", "nan", "NaN") else float("nan")
                    n_vox = int(float(raw_n)) if raw_n not in ("", "nan", "NaN") else 0
                except ValueError:
                    value, n_vox = float("nan"), 0

                records.append({
                    "block_id":    block_id,
                    "metric":      metric,
                    "roi_raw":     roi_raw,
                    "patch_index": patch_index,
                    "value":       value,
                    "n_voxels":    n_vox,
                })

    log.info(f"Loaded {len(records)} records from {data_dir}")
    return records


def pivot_wide(records, roi_lookup):
    """One row per (block_id, roi_type, patch_index); metrics become columns."""
    key_map = defaultdict(dict)
    for rec in records:
        key = (rec["block_id"], roi_lookup[rec["roi_raw"]], rec["patch_index"])
        key_map[key][rec["metric"]]               = rec["value"]
        key_map[key][rec["metric"] + "_n_voxels"] = rec["n_voxels"]

    rows = []
    for (block_id, roi_type, patch_index), vals in key_map.items():
        row = {"block_id": block_id, "roi_type": roi_type, "patch_index": patch_index}
        for met in METRICS:
            row[met]               = vals.get(met, float("nan"))
            row[met + "_n_voxels"] = vals.get(met + "_n_voxels", 0)
        rows.append(row)

    before = len(rows)
    rows = [r for r in rows if not (math.isnan(r["FA"]) and math.isnan(r["MD"]))]
    log.info(f"Dropped {before - len(rows)} subpatches where both FA and MD are missing")
    log.info(f"Wide dataset: {len(rows)} subpatches across "
             f"{len(set(r['block_id'] for r in rows))} blocks, "
             f"{len(set(r['roi_type'] for r in rows))} ROI types")
    return rows


def compute_normalization_params(rows):
    params = {}
    for col in ZSCORE_COLS:
        mu, sigma = mean_std([r[col] for r in rows if not math.isnan(r[col])])
        params[col] = {"transform": "none", "mu": mu, "sigma": sigma}

    for col in LOG_ZSCORE_COLS:
        raw_vals = [r[col] for r in rows if not math.isnan(r[col]) and r[col] >= 0]
        eps = percentile([v for v in raw_vals if v > 0], 5)
        mu, sigma = mean_std([math.log10(v + eps) for v in raw_vals])
        params[col] = {"transform": "log10", "eps": eps, "mu": mu, "sigma": sigma}
        log.info(f"  {col}: eps={eps:.6g}")
    return params


def apply_normalization(rows, params):
    """Add {col}_z for every normalized column; raw columns are kept."""
    for row in rows:
        for col, p in params.items():
            raw = row[col]
            if math.isnan(raw):
                row[col + "_z"] = float("nan")
            elif p["transform"] == "log10":
                row[col + "_z"] = (math.log10(max(raw, 0) + p["eps"]) - p["mu"]) / p["sigma"]
            else:
                row[col + "_z"] = (raw - p["mu"]) / p["sigma"]
    return rows


def add_derived_columns(rows):
    # Patch-size term for the residual scale: log(n_voxels / median n_voxels)
    fa_counts = [r["FA_n_voxels"] for r in rows if r["FA_n_voxels"] > 0]
    n_ref = sorted(fa_counts)[len(fa_counts) // 2]
    log.info(f"n_voxels reference (median FA voxel count): {n_ref}")
    for row in rows:
        nv = row["FA_n_voxels"] if row["FA_n_voxels"] > 0 else n_ref
        row["log_n_ratio"] = math.log(nv / n_ref)

    roi_types = sorted(set(r["roi_type"] for r in rows))
    blocks    = sorted(set(r["block_id"] for r in rows))
    roi_idx   = {v: i for i, v in enumerate(roi_types)}
    block_idx = {v: i for i, v in enumerate(blocks)}
    for row in rows:
        row["roi_idx"]   = roi_idx[row["roi_type"]]
        row["block_idx"] = block_idx[row["block_id"]]

    index_maps = {"roi_types": roi_types, "blocks": blocks, "n_ref_voxels": n_ref}
    return rows, index_maps


def make_sample_size_table(rows, index_maps):
    blocks    = index_maps["blocks"]
    roi_types = index_maps["roi_types"]
    counts    = defaultdict(lambda: defaultdict(int))
    for row in rows:
        counts[row["roi_type"]][row["block_id"]] += 1

    table_rows = [["roi_type"] + blocks + ["total"]]
    for roi in roi_types:
        row_vals = [roi] + [str(counts[roi].get(b, 0)) for b in blocks]
        row_vals.append(str(sum(counts[roi].get(b, 0) for b in blocks)))
        table_rows.append(row_vals)
    totals = ["TOTAL"] + [str(sum(counts[roi].get(b, 0) for roi in roi_types)) for b in blocks]
    totals.append(str(sum(int(v) for v in totals[1:])))
    table_rows.append(totals)
    return table_rows


def main():
    OUT_DIR.mkdir(exist_ok=True)

    pooling    = load_yaml(CONFIG_DIR / "roi_pooling.yaml")
    block_meta = load_yaml(CONFIG_DIR / "block_metadata.yaml")
    roi_lookup = {raw: pooled for pooled, members in pooling.items() for raw in members}

    records = load_all_csvs(DATA_DIR)
    rows = pivot_wide(records, roi_lookup)
    for row in rows:
        for col in TECH_COLS:
            row[col] = float(block_meta[row["block_id"]][col])

    params = compute_normalization_params(rows)
    rows = apply_normalization(rows, params)
    rows, index_maps = add_derived_columns(rows)

    with open(OUT_DIR / "normalization_params.json", "w") as f:
        json.dump(params, f, indent=2)
    with open(OUT_DIR / "index_maps.json", "w") as f:
        json.dump(index_maps, f, indent=2)
    with open(OUT_DIR / "analysis_dataset.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(OUT_DIR / "sample_size_table.csv", "w", newline="") as f:
        csv.writer(f).writerows(make_sample_size_table(rows, index_maps))

    log.info(f"Saved {len(rows)} subpatches to {OUT_DIR}")


if __name__ == "__main__":
    main()
