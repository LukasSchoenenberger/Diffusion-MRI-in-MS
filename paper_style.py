"""
paper_style.py
==============
Shared labels, colours, style and posterior loaders for the manuscript figures
and Table 4 (07-11). Pipeline variant tags map to the manuscript names:
    B  = regional analysis
    C1 = G2 analysis (pooled slope, single global intercept)
    C2 = G1 analysis (pooled slope, ROI-specific intercepts)
"""

import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
IDX_FILE = BASE_DIR / "Analysis_Dataset" / "index_maps.json"

CM_PER_INCH = 2.54
HDI_PROB = 0.95

# ROI types represented in fewer than six blocks; modelled but not reported.
EXCLUDE_ROIS = {"Granular", "Molecular", "dirtyWM", "Repair"}

ROI_LABEL = {
    "WM": "NAWM",
    "Surrounding_WM": "SWM",
    "WM_partial_demyel": "PDWM",
    "Perilesion": "PL",
    "Lesion_core": "LC",
    "Cortex_1": "Cortex I",
    "Cortex_2": "Cortex II",
    "Cortex_3": "Cortex III",
    "Cortex_diffuse_demyel": "DDC",
    "Cortex_partial_demyel": "PDC",
}

# Manuscript order: white matter -> lesion -> cortex.
ROI_ORDER = [
    "WM",
    "Surrounding_WM",
    "WM_partial_demyel",
    "Perilesion",
    "Lesion_core",
    "Cortex_1",
    "Cortex_2",
    "Cortex_3",
    "Cortex_diffuse_demyel",
    "Cortex_partial_demyel",
]

EXPOSURE_LABEL = {
    "F1": "Axon Density",
    "F2": "Fiber Coherence",
    "F3": "Myelin Density",
    "F4": "Astrocyte Density",
    "F5": "Microglia Density",
    "F6": "Cell Density",
    "F7": "Axon Density CDE",
    "F8": "Astrocyte Density CDE",
    "F9": "Microglia Density CDE",
    "M1": "Axon Density",
    "M2": "Myelin Density",
    "M3": "Astrocyte Density",
    "M4": "Microglia Density",
    "M5": "Cell Density",
    "M6": "Axon Density CDE",
    "M7": "Astrocyte Density CDE",
    "M8": "Microglia Density CDE",
}

MODEL_LABEL = {
    mid: f"{label} -> {'FA' if mid.startswith('F') else 'MD'}"
    for mid, label in EXPOSURE_LABEL.items()
}

EXPOSURE_SHORT = {
    mid: label.replace(" Density", "").replace("Fiber ", "")
    for mid, label in EXPOSURE_LABEL.items()
}

MEDIATION_LABEL = {
    "cde": "CDE",
    "indirect_myelin": "via Myelin Density",
    "indirect_coherence": "via Fiber Coherence",
    "indirect_cell": "via Cell Density",
    "total": "Total",
}

VARIANT_LABEL = {"B": "Regional analysis", "C1": "G2 analysis", "C2": "G1 analysis"}

FA_MODELS = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"]
MD_MODELS = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]

COL = {
    "pos_credible": "#D55E00",
    "neg_credible": "#0072B2",
    "uncertain": "#999999",
    "primary": "#0072B2",
    "cde": "#0072B2",
    "indirect_myelin": "#E69F00",
    "indirect_coherence": "#009E73",
    "indirect_cell": "#CC79A7",
    "total": "#555555",
}

ROI_COLORS = {
    "WM": "#0072B2",
    "Surrounding_WM": "#56B4E9",
    "WM_partial_demyel": "#009E73",
    "Perilesion": "#E69F00",
    "Lesion_core": "#D55E00",
    "Cortex_1": "#CC79A7",
    "Cortex_2": "#44AA99",
    "Cortex_3": "#332288",
    "Cortex_diffuse_demyel": "#882255",
    "Cortex_partial_demyel": "#DDAA33",
}


def setup_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "DejaVu Serif",
            ],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 2000,
        }
    )


def rlabel(roi: str) -> str:
    return ROI_LABEL.get(roi, roi)


def block_order() -> list[str]:
    """Internal block IDs in ascending numeric order (the B1..B13 order)."""
    with IDX_FILE.open() as handle:
        blocks = json.load(handle)["blocks"]
    return sorted(blocks, key=lambda b: int(b.rsplit("_", 1)[1]))


def block_labels() -> dict[str, str]:
    """Internal block ID -> manuscript label B1..B13."""
    return {block: f"B{n}" for n, block in enumerate(block_order(), start=1)}


def blabel(block: str) -> str:
    return block_labels().get(block, block)


def ordered_rois() -> list[str]:
    """Reported ROI types in manuscript order."""
    return list(ROI_ORDER)


def effect_color(p_gt0: float) -> str:
    # pd >= 0.975 is the credibility convention used in the figures and tables.
    if p_gt0 >= 0.975:
        return COL["pos_credible"]
    if p_gt0 <= 0.025:
        return COL["neg_credible"]
    return COL["uncertain"]


def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the ArviZ-style interval columns (hdi_2.5%, hdi_97.5%)."""
    return df.rename(columns={"hdi_2.5%": "hdi_2.5", "hdi_97.5%": "hdi_97.5"})


def indexed_rois() -> list[str]:
    """All 14 ROI types in the index order of beta_exp and roi_idx."""
    with IDX_FILE.open() as handle:
        return json.load(handle)["roi_types"]


def hdi(draws: np.ndarray, prob: float = HDI_PROB) -> tuple[float, float]:
    """Shortest interval containing `prob` of the draws (matches arviz.hdi)."""
    ordered = np.sort(np.asarray(draws, dtype=float))
    n = ordered.size
    width = int(np.floor(prob * n))
    spans = ordered[width:] - ordered[: n - width]
    start = int(np.argmin(spans))
    return float(ordered[start]), float(ordered[start + width])


def summarize_draws(draws: np.ndarray) -> dict[str, float]:
    lo, hi = hdi(draws)
    return {
        "mean": float(np.mean(draws)),
        "hdi_2.5": lo,
        "hdi_97.5": hi,
        "P(>0)": float(np.mean(np.asarray(draws) > 0)),
    }


_BETA_CACHE: dict[tuple[str, str, str, str], np.ndarray] = {}


def load_beta_exp_draws(results_dir: Path, outcome: str, model_id: str, variant: str = "B") -> np.ndarray:
    """Target-exposure posterior draws: (n_draws, n_roi) for B, (n_draws,) for C1/C2."""
    key = (str(results_dir), outcome, model_id, variant)
    if key not in _BETA_CACHE:
        path = results_dir / f"{outcome}_{variant}" / f"{model_id}_idata.nc"
        with h5py.File(path, "r") as handle:
            draws = np.asarray(handle["posterior"]["beta_exp"][...], dtype=float)
        _BETA_CACHE[key] = draws.reshape(-1, draws.shape[-1]) if draws.ndim == 3 else draws.reshape(-1)
    return _BETA_CACHE[key]


def load_regional_table(results_dir: Path, outcome: str) -> pd.DataFrame:
    """Per-ROI effect summaries (mean, 95% HDI, P(>0)) from the regional posteriors."""
    all_rois = indexed_rois()
    models = FA_MODELS if outcome == "FA" else MD_MODELS
    rows = []
    for model_id in models:
        draws = load_beta_exp_draws(results_dir, outcome, model_id)
        for idx, roi in enumerate(all_rois):
            if roi in EXCLUDE_ROIS:
                continue
            rows.append({"model": model_id, "roi_type": roi, **summarize_draws(draws[:, idx])})
    return pd.DataFrame(rows)


def load_estimand_table(results_dir: Path, outcome: str, variant: str) -> pd.DataFrame:
    """Pooled-slope (C1/C2) effect summaries from the posteriors."""
    models = FA_MODELS if outcome == "FA" else MD_MODELS
    rows = [
        {"model": model_id, **summarize_draws(load_beta_exp_draws(results_dir, outcome, model_id, variant))}
        for model_id in models
    ]
    return pd.DataFrame(rows)


def load_gcomp_table(results_dir: Path, outcome: str) -> pd.DataFrame:
    """Per-ROI g-computation summary of F10 (FA) / M9 (MD), regional analysis."""
    model = "F10" if outcome == "FA" else "M9"
    df = norm_cols(pd.read_csv(results_dir / f"{outcome}_B" / f"{model}_gcomputation_by_roi_summary.csv"))
    return df[~df["roi"].isin(EXCLUDE_ROIS)].copy()
