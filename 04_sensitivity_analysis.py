"""
04_sensitivity_analysis.py
==========================
Sensitivity analyses and diagnostics.

Regional analysis (B):
  S1   Student-t likelihood (nu = 4)                   F3, F6, F7, M2, M5, M6
  S2a  Fixed-effect priors N(0, 2.5)                   F3, F6, F7, M2, M5, M6
  S2b  Fixed-effect priors N(0, 5)                     F3, F6, F7, M2, M5, M6
  S3d  sqrt instead of log10 transform of microglia    F5, F6, M4, M5
  S4   No patch-size weighting (delta_n = 0)           F3, F6, F7, M2, M5, M6
  S6   Block as fixed effect                           F3, M2
  S7   Leave-one-block-out                             F3, M2
  Proxy-validity simulation for the latent oligodendrocyte confounder (F7)
  Diagnostics: VIF and within-ROI predictor correlations, E-values, Imai rho
  sensitivity and drop-one-mediator g-computation for F10/M9, Moran's I.

All enabled variants (B, C1, C2):
  S5   Heteroscedastic vs homoscedastic residuals, PSIS-LOO   F1, F3, F7

Requires the fitted models of 02_FA_models.py and 03_MD_models.py.

Outputs in Results/Sensitivity_<variant>/:
  {model}_{sensitivity}_idata.nc, {model}_{sensitivity}_summary.csv
  tier1_summary.csv, tier2_looic.csv, lobo_summary.csv,
  item12_proxy_validity_summary.csv, vif_summary.csv, within_roi_correlations.csv,
  evalues_summary.csv, imai_rho_sensitivity.csv, drop_one_mediator.csv,
  spatial_correlation_morans_i.csv
"""

import copy
import csv
import json
import logging
import math
import tempfile
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent
DATA_FILE = BASE_DIR / "Analysis_Dataset" / "analysis_dataset.csv"
IDX_FILE  = BASE_DIR / "Analysis_Dataset" / "index_maps.json"
SPEC_FILE = BASE_DIR / "config" / "model_specs.yaml"
RUN_CTRL  = BASE_DIR / "config" / "run_control.yaml"

TIER1_FA_MODELS = ["F3", "F6", "F7"]
TIER1_MD_MODELS = ["M2", "M5", "M6"]
S3D_MODELS      = {"F5", "M4", "F6", "M5"}

TECH_COLS = ("FixDur_days_z", "StorTime_days_z")

N_DRAWS       = 2000
N_TUNE        = 1000
N_CHAINS      = 4
TARGET_ACCEPT = 0.9
RANDOM_SEED   = 42


def load_dataset(data_file):
    def flt(v):
        try:
            x = float(v)
            return x if not math.isnan(x) else None
        except (ValueError, TypeError):
            return None

    rows = []
    with open(data_file, newline="") as f:
        for r in csv.DictReader(f):
            row = {k: flt(v) if k not in ("block_id", "roi_type") else v for k, v in r.items()}
            row["roi_idx"]   = int(r["roi_idx"])
            row["block_idx"] = int(r["block_idx"])
            rows.append(row)
    return rows


def build_arrays(rows, exposure_col, adj_cols, outcome_col, block_subset=None):
    """Complete-case design arrays; block_subset restricts the rows (LOBO)."""
    if block_subset is not None:
        rows = [r for r in rows if r["block_id"] in block_subset]

    required = [outcome_col, exposure_col] + adj_cols
    sel = [r for r in rows if all(r.get(c) is not None for c in required)]
    n_dropped = len(rows) - len(sel)
    if n_dropped:
        log.info(f"  Complete-case: dropped {n_dropped}/{len(rows)} rows")

    X_adj = np.zeros((len(sel), len(adj_cols)), dtype=float)
    for j, col in enumerate(adj_cols):
        X_adj[:, j] = [r[col + "_z"] for r in sel]

    X_tech = np.zeros((len(sel), len(TECH_COLS)), dtype=float)
    for j, col in enumerate(TECH_COLS):
        X_tech[:, j] = [r[col] for r in sel]

    return dict(
        y=np.array([r[outcome_col] for r in sel], dtype=float),
        X_exp=np.array([r[exposure_col + "_z"] for r in sel], dtype=float),
        X_adj=X_adj,
        X_tech=X_tech,
        roi_idx=np.array([r["roi_idx"] for r in sel], dtype=int),
        block_idx=np.array([r["block_idx"] for r in sel], dtype=int),
        log_n=np.array([r["log_n_ratio"] for r in sel], dtype=float),
        n=len(sel),
        adj_cols=adj_cols,
    )


def sqrt_microglia_rows(rows):
    """Copy of rows with microglia_density_z from sqrt(x) + global z-score (S3d)."""
    rows_alt  = copy.deepcopy(rows)
    sqrt_vals = [math.sqrt(max(r.get("microglia_density") or 0.0, 0.0)) for r in rows_alt]
    n     = len(sqrt_vals)
    mu    = sum(sqrt_vals) / n
    sigma = math.sqrt(sum((v - mu) ** 2 for v in sqrt_vals) / (n - 1))
    for r, sv in zip(rows_alt, sqrt_vals):
        r["microglia_density_z"] = (sv - mu) / sigma
    return rows_alt


def hdi_95(draws):
    lo, hi = az.hdi(np.asarray(draws, dtype=float), hdi_prob=0.95)
    return float(lo), float(hi)


def write_csv(path, records):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    log.info(f"Saved {path} ({len(records)} rows)")


# ---------------------------------------------------------------------------
# Sensitivity model builders (regional structure of 02/03 unless noted)
# ---------------------------------------------------------------------------
def _coords(data, index_maps):
    coords = {"roi": index_maps["roi_types"], "block": index_maps["blocks"]}
    if data["X_adj"].shape[1] > 0:
        coords["adj"] = data["adj_cols"]
    return coords


def _adj_effect(data, prior_sigma=1.0, global_pooled=False):
    if data["X_adj"].shape[1] == 0:
        return 0.0
    if global_pooled:
        beta_adj = pm.Normal("beta_adj", 0, prior_sigma, dims="adj")
        return pm.math.dot(data["X_adj"], beta_adj)
    beta_adj = pm.Normal("beta_adj", 0, prior_sigma, dims=("roi", "adj"))
    return (beta_adj[data["roi_idx"]] * data["X_adj"]).sum(axis=-1)


def _likelihood(outcome_col, mu, sigma, data):
    """Skew-normal with partially pooled ROI skewness for MD, Normal for FA."""
    obs_name = outcome_col.replace("_z", "_obs")
    if outcome_col.startswith("MD"):
        mu_alpha_skew    = pm.Normal("mu_alpha_skew", 0, 1.0)
        sigma_alpha_skew = pm.HalfNormal("sigma_alpha_skew", 0.5)
        alpha_skew       = pm.Normal("alpha_skew", mu=mu_alpha_skew,
                                     sigma=sigma_alpha_skew, dims="roi")
        pm.SkewNormal(obs_name, mu=mu, sigma=sigma,
                      alpha=alpha_skew[data["roi_idx"]], observed=data["y"])
    else:
        pm.Normal(obs_name, mu=mu, sigma=sigma, observed=data["y"])


def build_model_student_t(data, index_maps, outcome_col, nu=4):
    """S1: Student-t likelihood."""
    with pm.Model(coords=_coords(data, index_maps)) as model:
        sigma_block = pm.HalfNormal("sigma_block", 1.0)
        v_block     = pm.Normal("v_block", 0, sigma_block, dims="block")
        alpha       = pm.Normal("alpha", 0, 1.0, dims="roi")
        beta_exp    = pm.Normal("beta_exp", 0, 1.0, dims="roi")
        adj_effect  = _adj_effect(data)
        tech_effect = pm.math.dot(data["X_tech"], pm.Normal("gamma", 0, 1.0, shape=len(TECH_COLS)))

        delta0        = pm.Normal("delta0", 0, 1.0)
        sigma_delta_r = pm.HalfNormal("sigma_delta_r", 0.5)
        delta_r       = pm.Normal("delta_r", 0, sigma_delta_r, dims="roi")
        delta_n       = pm.Normal("delta_n", -0.25, 0.5)
        sigma_eff = pm.Deterministic("sigma_eff", pm.math.exp(
            delta0 + delta_r[data["roi_idx"]] + delta_n * data["log_n"]))

        mu = (alpha[data["roi_idx"]] + beta_exp[data["roi_idx"]] * data["X_exp"]
              + adj_effect + tech_effect + v_block[data["block_idx"]])

        pm.StudentT(outcome_col.replace("_z", "_obs"), nu=nu,
                    mu=mu, sigma=sigma_eff, observed=data["y"])
    return model


def build_model_wide_prior(data, index_maps, prior_sigma, outcome_col):
    """S2a/S2b: wider fixed-effect priors; with prior_sigma=1 the primary model (S3d)."""
    with pm.Model(coords=_coords(data, index_maps)) as model:
        sigma_block = pm.HalfNormal("sigma_block", 1.0)
        v_block     = pm.Normal("v_block", 0, sigma_block, dims="block")
        alpha       = pm.Normal("alpha", 0, prior_sigma, dims="roi")
        beta_exp    = pm.Normal("beta_exp", 0, prior_sigma, dims="roi")
        adj_effect  = _adj_effect(data, prior_sigma)
        tech_effect = pm.math.dot(data["X_tech"],
                                  pm.Normal("gamma", 0, prior_sigma, shape=len(TECH_COLS)))

        delta0        = pm.Normal("delta0", 0, 1.0)
        sigma_delta_r = pm.HalfNormal("sigma_delta_r", 0.5)
        delta_r       = pm.Normal("delta_r", 0, sigma_delta_r, dims="roi")
        delta_n       = pm.Normal("delta_n", -0.25, 0.5)
        sigma_eff = pm.Deterministic("sigma_eff", pm.math.exp(
            delta0 + delta_r[data["roi_idx"]] + delta_n * data["log_n"]))

        mu = (alpha[data["roi_idx"]] + beta_exp[data["roi_idx"]] * data["X_exp"]
              + adj_effect + tech_effect + v_block[data["block_idx"]])

        _likelihood(outcome_col, mu, sigma_eff, data)
    return model


def build_model_fixed_delta_n(data, index_maps, outcome_col):
    """S4: residual scale without the patch-size term (delta_n = 0).
    Also used for the proxy-validity simulation, whose data have no technical covariates."""
    with pm.Model(coords=_coords(data, index_maps)) as model:
        sigma_block = pm.HalfNormal("sigma_block", 1.0)
        v_block     = pm.Normal("v_block", 0, sigma_block, dims="block")
        alpha       = pm.Normal("alpha", 0, 1.0, dims="roi")
        beta_exp    = pm.Normal("beta_exp", 0, 1.0, dims="roi")
        adj_effect  = _adj_effect(data)
        if data["X_tech"] is not None:
            tech_effect = pm.math.dot(data["X_tech"],
                                      pm.Normal("gamma", 0, 1.0, shape=len(TECH_COLS)))
        else:
            tech_effect = 0.0

        delta0        = pm.Normal("delta0", 0, 1.0)
        sigma_delta_r = pm.HalfNormal("sigma_delta_r", 0.5)
        delta_r       = pm.Normal("delta_r", 0, sigma_delta_r, dims="roi")
        sigma_eff = pm.Deterministic("sigma_eff", pm.math.exp(delta0 + delta_r[data["roi_idx"]]))

        mu = (alpha[data["roi_idx"]] + beta_exp[data["roi_idx"]] * data["X_exp"]
              + adj_effect + tech_effect + v_block[data["block_idx"]])

        _likelihood(outcome_col, mu, sigma_eff, data)
    return model


def build_model_homoscedastic(data, index_maps, outcome_col, *,
                              global_pooled=False, include_roi_intercept=True):
    """S5: homoscedastic residuals; variant arguments as in 02/03."""
    with pm.Model(coords=_coords(data, index_maps)) as model:
        sigma_block = pm.HalfNormal("sigma_block", 1.0)
        v_block     = pm.Normal("v_block", 0, sigma_block, dims="block")

        if include_roi_intercept:
            alpha      = pm.Normal("alpha", 0, 1.0, dims="roi")
            alpha_term = alpha[data["roi_idx"]]
        else:
            alpha_term = pm.Normal("alpha0", 0, 1.0)

        if global_pooled:
            beta_exp      = pm.Normal("beta_exp", 0, 1.0)
            beta_exp_term = beta_exp * data["X_exp"]
        else:
            beta_exp      = pm.Normal("beta_exp", 0, 1.0, dims="roi")
            beta_exp_term = beta_exp[data["roi_idx"]] * data["X_exp"]

        adj_effect  = _adj_effect(data, global_pooled=global_pooled)
        tech_effect = pm.math.dot(data["X_tech"], pm.Normal("gamma", 0, 1.0, shape=len(TECH_COLS)))

        sigma = pm.HalfNormal("sigma", 1.0)

        mu = alpha_term + beta_exp_term + adj_effect + tech_effect + v_block[data["block_idx"]]

        _likelihood(outcome_col, mu, sigma, data)
    return model


def build_model_block_fixed(rows, exposure, adj_cols, index_maps, outcome_col):
    """S6: block as fixed effect instead of random intercept."""
    sel = [r for r in rows if all(r.get(c) is not None
                                  for c in [outcome_col, exposure] + adj_cols)]
    blocks = sorted(set(r["block_id"] for r in sel))
    n_adj  = len(adj_cols)
    coords = {"roi": index_maps["roi_types"], "block": blocks}
    if n_adj > 0:
        coords["adj"] = adj_cols

    block_lookup = {b: i for i, b in enumerate(blocks)}
    bidx    = np.array([block_lookup[r["block_id"]] for r in sel], dtype=int)
    y       = np.array([r[outcome_col] for r in sel], dtype=float)
    X_exp   = np.array([r[exposure + "_z"] for r in sel], dtype=float)
    roi_idx = np.array([r["roi_idx"] for r in sel], dtype=int)
    log_n   = np.array([r["log_n_ratio"] for r in sel], dtype=float)
    X_adj   = np.zeros((len(sel), n_adj), dtype=float)
    for j, col in enumerate(adj_cols):
        X_adj[:, j] = [r[col + "_z"] for r in sel]
    X_tech = np.zeros((len(sel), len(TECH_COLS)), dtype=float)
    for j, col in enumerate(TECH_COLS):
        X_tech[:, j] = [r[col] for r in sel]

    with pm.Model(coords=coords) as model:
        block_fe = pm.Normal("block_fe", 0, 1.0, shape=len(blocks))
        alpha    = pm.Normal("alpha", 0, 1.0, dims="roi")
        beta_exp = pm.Normal("beta_exp", 0, 1.0, dims="roi")

        if n_adj > 0:
            beta_adj   = pm.Normal("beta_adj", 0, 1.0, dims=("roi", "adj"))
            adj_effect = (beta_adj[roi_idx] * X_adj).sum(axis=-1)
        else:
            adj_effect = 0.0

        delta0        = pm.Normal("delta0", 0, 1.0)
        sigma_delta_r = pm.HalfNormal("sigma_delta_r", 0.5)
        delta_r       = pm.Normal("delta_r", 0, sigma_delta_r, dims="roi")
        delta_n       = pm.Normal("delta_n", -0.25, 0.5)
        sigma_eff = pm.Deterministic("sigma_eff", pm.math.exp(
            delta0 + delta_r[roi_idx] + delta_n * log_n))

        gamma       = pm.Normal("gamma", 0, 1.0, shape=X_tech.shape[1])
        tech_effect = (X_tech * gamma).sum(axis=-1)

        mu = alpha[roi_idx] + beta_exp[roi_idx] * X_exp + adj_effect + tech_effect + block_fe[bidx]

        _likelihood(outcome_col, mu, sigma_eff, {"roi_idx": roi_idx, "y": y})
    return model


def quick_sample(model, model_id, result_dir):
    nc_path = result_dir / f"{model_id}_idata.nc"
    if nc_path.exists():
        log.info(f"  {model_id}: loading existing {nc_path.name}")
        return az.from_netcdf(str(nc_path))

    with model:
        idata = pm.sample(draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS,
                          target_accept=TARGET_ACCEPT, random_seed=RANDOM_SEED,
                          idata_kwargs={"log_likelihood": True},
                          progressbar=True)
    idata.to_netcdf(str(nc_path))

    summary = az.summary(idata, var_names=["beta_exp"], hdi_prob=0.95)
    write_csv(result_dir / f"{model_id}_summary.csv",
              [{"parameter": p, **summary.loc[p].to_dict()} for p in summary.index])

    rhat = az.rhat(idata)
    divs = int(idata.sample_stats.diverging.values.sum())
    rhat_max = float(max(rhat[v].values.max() for v in rhat.data_vars))
    log.info(f"  {model_id}: divs={divs}, rhat_max={rhat_max:.4f}")
    return idata


def beta_exp_summary(idata, roi_types):
    be = idata.posterior["beta_exp"].values.reshape(-1, len(roi_types))
    rows = []
    for i, roi in enumerate(roi_types):
        lo, hi = hdi_95(be[:, i])
        rows.append({
            "roi":      roi,
            "mean":     float(be[:, i].mean()),
            "sd":       float(be[:, i].std()),
            "hdi_2.5":  lo,
            "hdi_97.5": hi,
            "P(>0)":    float((be[:, i] > 0).mean()),
        })
    return rows


# ---------------------------------------------------------------------------
# Proxy-validity simulation
# ---------------------------------------------------------------------------
# True Axon -> FA CDEs: F7 posterior means (regional analysis), z-scored scale
PROXY_TARGET_ROIS = {
    "WM":                -0.005,
    "WM_partial_demyel":  0.108,
    "Lesion_core":       -0.299,
}

DGP_BETA_MF  = 0.35   # Myelin -> FA
DGP_BETA_CF  = 0.20   # Cell -> FA
DGP_GAMMA_OC = 0.50   # Oligo-Cell correlation
DGP_SIGMA_FA = 0.50   # FA residual SD

GAMMA_OM_PROXY_GRID = [0.3, 0.5, 0.7, 0.9]         # grid 1, effect scale 1.0
BETA_SCALE_GRID     = [0.5, 0.75, 1.0, 1.25, 1.5]  # grid 2, gamma_om = 0.6
GAMMA_OM_BETA_GRID  = 0.6


def dgp_raw_betas(target_z, gamma_om):
    """
    Raw Axon -> FA slopes that give the target z-scored CDEs.

    With Var(Myelin) = Var(Cell) = 1 and Cov(Myelin, Cell) = gamma_om * gamma_oc:
      K        = beta_mf^2 + beta_cf^2 + 2 beta_mf beta_cf gamma_om gamma_oc + sigma_fa^2
      sigma_FA = sqrt(K / (1 - mean(target_z^2)))
      beta[r]  = target_z[r] * sigma_FA
    """
    t = np.array(list(target_z.values()))
    K = (DGP_BETA_MF**2
         + DGP_BETA_CF**2
         + 2.0 * DGP_BETA_MF * DGP_BETA_CF * gamma_om * DGP_GAMMA_OC
         + DGP_SIGMA_FA**2)
    sigma_fa = float(np.sqrt(K / (1.0 - float(np.mean(t**2)))))
    return {roi: tz * sigma_fa for roi, tz in target_z.items()}, sigma_fa


def run_proxy_grid_cell(gamma_om, beta_scale, result_dir, grid_label):
    """
    Simulate one synthetic dataset, fit the F7-equivalent model and return
    per-ROI bias and interval containment.

    DGP (standardized):
      Oligo ~ N(0,1) [latent], Axon ~ N(0,1)
      Myelin = gamma_om * Oligo + sqrt(1 - gamma_om^2) * e_M
      Cell   = gamma_oc * Oligo + sqrt(1 - gamma_oc^2) * e_C
      GFAP, Microglia, Coherence ~ N(0,1)
      FA[r]  = beta_axon[r] * Axon + beta_mf * Myelin + beta_cf * Cell + N(0, sigma_fa)
    """
    rng = np.random.default_rng(RANDOM_SEED)

    N_OBS_PER_ROI = 600
    N_BLOCKS      = 4
    roi_names     = list(PROXY_TARGET_ROIS.keys())
    N_ROI         = len(roi_names)
    N_OBS         = N_OBS_PER_ROI * N_ROI

    scaled_targets = {r: v * beta_scale for r, v in PROXY_TARGET_ROIS.items()}
    raw_betas, sigma_fa = dgp_raw_betas(scaled_targets, gamma_om)

    Oligo  = rng.normal(0, 1, N_OBS)
    Axon   = rng.normal(0, 1, N_OBS)
    Myelin = gamma_om * Oligo + np.sqrt(1.0 - gamma_om**2) * rng.normal(0, 1, N_OBS)
    Cell   = DGP_GAMMA_OC * Oligo + np.sqrt(1.0 - DGP_GAMMA_OC**2) * rng.normal(0, 1, N_OBS)
    GFAP   = rng.normal(0, 1, N_OBS)
    Micro  = rng.normal(0, 1, N_OBS)
    Coher  = rng.normal(0, 1, N_OBS)

    roi_idx  = np.repeat(np.arange(N_ROI), N_OBS_PER_ROI)
    beta_arr = np.array([raw_betas[r] for r in roi_names])
    FA_raw   = (beta_arr[roi_idx] * Axon
                + DGP_BETA_MF * Myelin
                + DGP_BETA_CF * Cell
                + rng.normal(0, DGP_SIGMA_FA, N_OBS))
    block_idx = rng.integers(0, N_BLOCKS, N_OBS)

    def gz(x):
        return (x - x.mean()) / (x.std() + 1e-9)

    data_syn = dict(
        y         = gz(FA_raw),
        X_exp     = gz(Axon),
        X_adj     = np.column_stack([gz(Myelin), gz(Coher), gz(Cell), gz(GFAP), gz(Micro)]),
        X_tech    = None,
        roi_idx   = roi_idx,
        block_idx = block_idx,
        log_n     = np.zeros(N_OBS),
        n         = N_OBS,
        adj_cols  = ["myelin_density", "axon_coherence", "cell_density",
                     "gfap_density", "microglia_density"],
    )
    idx_syn = {"roi_types": roi_names, "blocks": [f"block_{i}" for i in range(N_BLOCKS)]}

    true_beta_z = {r: raw_betas[r] / sigma_fa for r in roi_names}
    log.info(f"  [{grid_label}]  gamma_om={gamma_om:.1f}  scale={beta_scale:.2f}  "
             f"true_z={[round(v, 3) for v in true_beta_z.values()]}")

    idata = quick_sample(build_model_fixed_delta_n(data_syn, idx_syn, "FA_z"), grid_label, result_dir)

    be   = idata.posterior["beta_exp"].values.reshape(-1, N_ROI)
    rows = []
    for i, roi in enumerate(roi_names):
        s      = be[:, i]
        mean_s = float(s.mean())
        lo, hi = hdi_95(s)
        true_z = true_beta_z[roi]
        bias   = mean_s - true_z
        # Relative bias is undefined for a near-zero true effect (WM)
        rel_bias = bias / abs(true_z) if abs(true_z) >= 0.02 else float("nan")
        rows.append({
            "grid":            grid_label.split("_")[0],
            "gamma_om":        gamma_om,
            "beta_axon_scale": beta_scale,
            "roi":             roi,
            "true_beta_z":     round(true_z, 4),
            "estimated_mean":  round(mean_s, 4),
            "estimated_sd":    round(float(s.std()), 4),
            "hdi_2.5":         round(lo, 4),
            "hdi_97.5":        round(hi, 4),
            "bias":            round(bias, 4),
            "relative_bias":   round(rel_bias, 4) if not math.isnan(rel_bias) else "NA",
            "covered":         int(lo <= true_z <= hi),
        })
    return rows


def run_proxy_validity(result_dir):
    log.info("\n=== Proxy-validity simulation ===")
    all_rows = []
    for gamma_om in GAMMA_OM_PROXY_GRID:
        all_rows.extend(run_proxy_grid_cell(gamma_om, 1.0, result_dir,
                                            grid_label=f"proxy_gom{gamma_om:.1f}"))
    for scale in BETA_SCALE_GRID:
        all_rows.extend(run_proxy_grid_cell(GAMMA_OM_BETA_GRID, scale, result_dir,
                                            grid_label=f"effect_s{scale:.2f}"))
    write_csv(result_dir / "item12_proxy_validity_summary.csv", all_rows)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def compute_vif(X_design):
    """Variance inflation factor of each column of an (N x p) design matrix."""
    n, p = X_design.shape
    vifs = np.zeros(p)
    for j in range(p):
        y_j     = X_design[:, j]
        X_j_int = np.column_stack([np.ones(n), np.delete(X_design, j, axis=1)])
        coef, _, _, _ = np.linalg.lstsq(X_j_int, y_j, rcond=None)
        ss_res = np.sum((y_j - X_j_int @ coef) ** 2)
        ss_tot = np.sum((y_j - y_j.mean()) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        vifs[j] = 1.0 / max(1.0 - r2, 1e-12)
    return vifs


def run_vif_collinearity(rows, specs, index_maps, result_dir):
    """VIF for every multi-predictor model and within-ROI pairwise predictor correlations."""
    log.info("\n=== Diagnostics: VIF / collinearity ===")
    roi_types    = index_maps["roi_types"]
    vif_records  = []
    corr_records = []

    for outcome_tag, outcome_col in [("FA", "FA_z"), ("MD", "MD_z")]:
        for model_id, spec in specs[outcome_tag].items():
            if spec["type"] != "regression":
                continue
            adj_cols  = spec.get("adjustment", [])
            pred_cols = [spec["exposure"]] + adj_cols
            data   = build_arrays(rows, spec["exposure"], adj_cols, outcome_col)
            X_full = np.column_stack([data["X_exp"].reshape(-1, 1), data["X_adj"]])
            if X_full.shape[1] < 2:
                continue

            vifs = compute_vif(X_full)
            for j, col in enumerate(pred_cols):
                vif_records.append({"model": model_id, "predictor": col,
                                    "VIF": float(vifs[j]), "VIF_gt_5": vifs[j] > 5.0})

            for r_i, roi in enumerate(roi_types):
                mask = data["roi_idx"] == r_i
                if mask.sum() < 3:
                    continue
                corr_mat = np.corrcoef(X_full[mask].T)
                for j in range(len(pred_cols)):
                    for k in range(j + 1, len(pred_cols)):
                        corr_records.append({
                            "model": model_id, "roi": roi,
                            "var1": pred_cols[j], "var2": pred_cols[k],
                            "pearson_r": float(corr_mat[j, k]),
                        })

    write_csv(result_dir / "vif_summary.csv", vif_records)
    write_csv(result_dir / "within_roi_correlations.csv", corr_records)


def compute_evalue(point_est, bound_near_null):
    """E-values (VanderWeele & Ding, 2017) for a standardized coefficient and
    for the interval bound closest to zero, using RR = exp(0.91 * |beta|)."""
    def _evalue_from_rr(rr):
        if rr < 1.0:
            rr = 1.0 / rr
        return rr + math.sqrt(rr * (rr - 1.0))

    ev_point = _evalue_from_rr(math.exp(0.91 * abs(point_est)))
    if abs(bound_near_null) < 1e-8:
        ev_bound = 1.0   # interval includes zero
    else:
        ev_bound = _evalue_from_rr(math.exp(0.91 * abs(bound_near_null)))
    return ev_point, ev_bound


def run_evalues(index_maps, result_dir, fa_dir, md_dir):
    """E-values for the total-effect estimands (point estimate and 95% HDI bound)."""
    log.info("\n=== Diagnostics: E-values ===")
    roi_types = index_maps["roi_types"]
    records   = []
    for res_dir, model_ids in [(fa_dir, ["F1", "F3", "F4", "F5"]),
                               (md_dir, ["M1", "M2", "M3", "M4"])]:
        for model_id in model_ids:
            idata = az.from_netcdf(str(res_dir / f"{model_id}_idata.nc"))
            be = idata.posterior["beta_exp"].values.reshape(-1, len(roi_types))
            for r_i, roi in enumerate(roi_types):
                draws    = be[:, r_i]
                mean_val = float(draws.mean())
                lo, hi   = hdi_95(draws)
                ev_point, ev_hdi = compute_evalue(mean_val, max(lo, 0.0) if mean_val > 0 else min(hi, 0.0))
                records.append({
                    "model": model_id, "roi": roi,
                    "beta_mean": mean_val,
                    "hdi_2.5": lo, "hdi_97.5": hi,
                    "evalue_point": ev_point,
                    "evalue_hdi": ev_hdi,
                })
    write_csv(result_dir / "evalues_summary.csv", records)


def posterior_sigma_by_roi(idata, n_roi):
    """Posterior mean residual SD per ROI at the reference patch size (log_n = 0)."""
    post = idata.posterior
    d0 = post["delta0"].values.reshape(-1)
    dr = post["delta_r"].values.reshape(-1, n_roi)
    return np.exp(d0[:, None] + dr).mean(axis=0)


def run_imai_rho_sensitivity(index_maps, result_dir, fa_dir, md_dir):
    """Imai, Keele & Yamamoto (2010) residual-correlation sensitivity:
        indirect(rho) = indirect - rho * sigma_M * sigma_Y
        rho_crit      = indirect / (sigma_M * sigma_Y)
    with ROI-level residual SDs of the mediator and CDE outcome models."""
    log.info("\n=== Diagnostics: Imai rho sensitivity ===")
    roi_types = index_maps["roi_types"]
    n_roi     = len(roi_types)
    rho_grid  = np.linspace(0.0, 0.95, 20)
    records   = []

    def add_records(outcome, component, roi, ie_mean, sigma_y, sigma_m):
        sp = float(sigma_y) * float(sigma_m)
        rho_crit = float(np.clip(ie_mean / sp, -1.0, 1.0))
        records.extend({
            "outcome":       outcome,
            "component":     component,
            "roi":           roi,
            "rho":           float(rho),
            "indirect_adj":  ie_mean - rho * sp,
            "indirect_obs":  ie_mean,
            "sigma_product": sp,
            "rho_crit":      rho_crit,
        } for rho in rho_grid)

    label = {"myelin_density": "myelin", "axon_coherence": "coherence", "cell_density": "cell"}
    for outcome, res_dir, gcomp_id, cde_id, total_id, mediators in [
        ("FA", fa_dir, "F10", "F7", "F1", ["myelin_density", "axon_coherence", "cell_density"]),
        ("MD", md_dir, "M9",  "M6", "M1", ["myelin_density", "cell_density"]),
    ]:
        idata_cde = az.from_netcdf(str(res_dir / f"{cde_id}_idata.nc"))
        sigma_y = posterior_sigma_by_roi(idata_cde, n_roi)
        sigma_m = {m: posterior_sigma_by_roi(
                       az.from_netcdf(str(res_dir / f"{gcomp_id}_mediator_{m}_idata.nc")), n_roi)
                   for m in mediators}

        # Total indirect effect: difference of coefficients (total - CDE)
        be_total = az.from_netcdf(str(res_dir / f"{total_id}_idata.nc")).posterior["beta_exp"].values
        be_cde   = idata_cde.posterior["beta_exp"].values
        indirect = be_total.reshape(-1, n_roi) - be_cde.reshape(-1, n_roi)
        for r_i, roi in enumerate(roi_types):
            add_records(outcome, f"total_indirect_{total_id}-{cde_id}", roi,
                        float(indirect[:, r_i].mean()), sigma_y[r_i], sigma_m["myelin_density"][r_i])

        # Mediator-specific indirect effects from the g-computation
        df_gc = pd.read_csv(res_dir / f"{gcomp_id}_gcomputation_by_roi_summary.csv")
        for m in mediators:
            comp_key = f"indirect_{label[m]}"
            for _, row in df_gc[df_gc["component"] == comp_key].iterrows():
                r_i = roi_types.index(row["roi"])
                add_records(outcome, f"gcomp_{comp_key}", str(row["roi"]),
                            float(row["mean"]), sigma_y[r_i], sigma_m[m][r_i])

    write_csv(result_dir / "imai_rho_sensitivity.csv", records)


def run_drop_one_mediator(rows, index_maps, result_dir, fa_dir, md_dir):
    """Re-run the F10/M9 g-computation with each mediator omitted in turn."""
    log.info("\n=== Diagnostics: drop-one-mediator ===")
    fa_mod  = import_module("02_FA_models")
    md_mod  = import_module("03_MD_models")
    records = []
    for gcomp_id, res_dir, cde_id, mediators, gcomputation in [
        ("F10", fa_dir, "F7", ["myelin_density", "axon_coherence", "cell_density"],
         fa_mod.gcomputation_F10),
        ("M9",  md_dir, "M6", ["myelin_density", "cell_density"],
         md_mod.gcomputation_M9),
    ]:
        idata_cde = az.from_netcdf(str(res_dir / f"{cde_id}_idata.nc"))
        for drop_med in mediators:
            log.info(f"  {gcomp_id}: dropping {drop_med}")
            mediator_idatas = {
                m: az.from_netcdf(str(res_dir / f"{gcomp_id}_mediator_{m}_idata.nc"))
                for m in mediators if m != drop_med
            }
            with tempfile.TemporaryDirectory() as tmpdir:
                gcomputation(rows, index_maps, idata_cde, mediator_idatas, Path(tmpdir))
                with open(Path(tmpdir) / f"{gcomp_id}_gcomputation_by_roi_summary.csv",
                          newline="") as f:
                    for row in csv.DictReader(f):
                        records.append({
                            "gcomp_model":      gcomp_id,
                            "dropped_mediator": drop_med,
                            "roi":              row["roi"],
                            "component":        row["component"],
                            "mean":             float(row["mean"]),
                            "sd":               float(row["sd"]),
                            "hdi_2.5%":         float(row["hdi_2.5%"]),
                            "hdi_97.5%":        float(row["hdi_97.5%"]),
                            "P(>0)":            float(row["P(>0)"]),
                        })
    write_csv(result_dir / "drop_one_mediator.csv", records)


def run_spatial_correlation(rows, specs, index_maps, result_dir, fa_dir, md_dir):
    """Moran's I of the posterior-mean residuals with same-block weights
    (w_ij = 1 if subpatches i and j are in the same block, 0 otherwise)."""
    log.info("\n=== Diagnostics: residual Moran's I ===")
    records = []
    for outcome_tag, outcome_col, res_dir in [("FA", "FA_z", fa_dir), ("MD", "MD_z", md_dir)]:
        for model_id, spec in specs[outcome_tag].items():
            nc_path = res_dir / f"{model_id}_idata.nc"
            if spec["type"] != "regression" or not nc_path.exists():
                continue
            idata   = az.from_netcdf(str(nc_path))
            obs_var = f"{outcome_tag}_obs"
            residuals = (idata.observed_data[obs_var].values
                         - idata.posterior_predictive[obs_var].values.mean(axis=(0, 1)))
            block_idx = build_arrays(rows, spec["exposure"], spec.get("adjustment", []),
                                     outcome_col)["block_idx"]

            n = len(residuals)
            r_centered = residuals - residuals.mean()
            w_sum = 0.0
            numerator = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    if block_idx[i] == block_idx[j]:
                        w_sum += 2.0
                        numerator += 2.0 * r_centered[i] * r_centered[j]

            morans_i = (n / w_sum) * (numerator / float(np.sum(r_centered ** 2)))
            e_i = -1.0 / (n - 1)
            records.append({
                "model": model_id,
                "morans_I": float(morans_i),
                "expected_I": float(e_i),
                "n_obs": n,
                "n_neighbor_pairs": int(w_sum / 2),
            })
            log.info(f"  {model_id}: Moran's I = {morans_i:.4f} (E[I] = {e_i:.4f})")
    write_csv(result_dir / "spatial_correlation_morans_i.csv", records)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def run_loo_comparison(rows, specs, index_maps, result_dir, fa_dir, *,
                       global_pooled=False, include_roi_intercept=True):
    """S5: primary (heteroscedastic) vs homoscedastic FA models, PSIS-LOO."""
    log.info("\n=== S5: heteroscedastic vs homoscedastic ===")
    loo_records = []
    for model_id in ["F1", "F3", "F7"]:
        spec = specs["FA"][model_id]
        data = build_arrays(rows, spec["exposure"], spec.get("adjustment", []), "FA_z")
        idata_het  = az.from_netcdf(str(fa_dir / f"{model_id}_idata.nc"))
        mdl_homo   = build_model_homoscedastic(data, index_maps, "FA_z",
                                               global_pooled=global_pooled,
                                               include_roi_intercept=include_roi_intercept)
        idata_homo = quick_sample(mdl_homo, f"{model_id}_S5_homoscedastic", result_dir)

        loo_het  = az.loo(idata_het,  pointwise=False)
        loo_homo = az.loo(idata_homo, pointwise=False)
        loo_records += [
            {"model": model_id, "variant": "heteroscedastic_B",
             "elpd_loo": float(loo_het.elpd_loo), "p_loo": float(loo_het.p_loo),
             "se": float(loo_het.se)},
            {"model": model_id, "variant": "homoscedastic_B",
             "elpd_loo": float(loo_homo.elpd_loo), "p_loo": float(loo_homo.p_loo),
             "se": float(loo_homo.se)},
        ]
    write_csv(result_dir / "tier2_looic.csv", loo_records)


def run_regional_analyses(rows, specs, index_maps, sens_flags, diag_flags,
                          result_dir, fa_dir, md_dir):
    roi_types = index_maps["roi_types"]
    blocks    = index_maps["blocks"]

    # Tier 1: S1, S2a, S2b, S3d, S4
    tier1_records = []

    def add_tier1(model_id, sensitivity, idata):
        for r in beta_exp_summary(idata, roi_types):
            tier1_records.append({"model": model_id, "sensitivity": sensitivity, **r})

    def run_s3d(model_id, exposure, adj_cols, outcome_col):
        sid  = f"{model_id}_S3d_sqrtMicroglia"
        log.info(f"\n--- {sid} ---")
        data = build_arrays(sqrt_microglia_rows(rows), exposure, adj_cols, outcome_col)
        add_tier1(model_id, "S3d_sqrtMicroglia",
                  quick_sample(build_model_wide_prior(data, index_maps, 1.0, outcome_col),
                               sid, result_dir))

    for outcome_tag, model_ids, outcome_col in [("FA", TIER1_FA_MODELS, "FA_z"),
                                                ("MD", TIER1_MD_MODELS, "MD_z")]:
        for model_id in model_ids:
            spec     = specs[outcome_tag][model_id]
            exposure = spec["exposure"]
            adj_cols = spec.get("adjustment", [])
            data     = build_arrays(rows, exposure, adj_cols, outcome_col)

            if sens_flags.get("S1_student_t", True):
                sid = f"{model_id}_S1_StudentT"
                log.info(f"\n--- {sid} ---")
                add_tier1(model_id, "S1_StudentT",
                          quick_sample(build_model_student_t(data, index_maps, outcome_col),
                                       sid, result_dir))

            for flag, tag, prior_sigma in [("S2a_prior_wide", "S2a_prior2.5", 2.5),
                                           ("S2b_prior_wider", "S2b_prior5.0", 5.0)]:
                if sens_flags.get(flag, True):
                    sid = f"{model_id}_{tag}"
                    log.info(f"\n--- {sid} ---")
                    add_tier1(model_id, tag,
                              quick_sample(build_model_wide_prior(data, index_maps, prior_sigma,
                                                                  outcome_col),
                                           sid, result_dir))

            if sens_flags.get("S3d_microglia_sqrt", True) and model_id in S3D_MODELS:
                run_s3d(model_id, exposure, adj_cols, outcome_col)

            if sens_flags.get("S4_delta_n_zero", True):
                sid = f"{model_id}_S4_noPatchWeight"
                log.info(f"\n--- {sid} ---")
                add_tier1(model_id, "S4_noPatchWeight",
                          quick_sample(build_model_fixed_delta_n(data, index_maps, outcome_col),
                                       sid, result_dir))

    # S3d for the microglia total-effect models outside the tier-1 set
    if sens_flags.get("S3d_microglia_sqrt", True):
        for model_id, outcome_tag, outcome_col in [("F5", "FA", "FA_z"), ("M4", "MD", "MD_z")]:
            spec = specs[outcome_tag][model_id]
            run_s3d(model_id, spec["exposure"], spec.get("adjustment", []), outcome_col)

    if tier1_records:
        write_csv(result_dir / "tier1_summary.csv", tier1_records)

    # S6: block fixed effect
    if sens_flags.get("S6_block_fixed", True):
        for model_id, outcome_tag, outcome_col in [("F3", "FA", "FA_z"), ("M2", "MD", "MD_z")]:
            spec = specs[outcome_tag][model_id]
            sid  = f"{model_id}_S6_blockFE"
            log.info(f"\n--- {sid} ---")
            quick_sample(build_model_block_fixed(rows, spec["exposure"],
                                                 spec.get("adjustment", []),
                                                 index_maps, outcome_col),
                         sid, result_dir)

    # S7: leave-one-block-out with the primary model builders of 02/03
    if sens_flags.get("S7_lobo", True):
        fa_mod = import_module("02_FA_models")
        md_mod = import_module("03_MD_models")
        lobo_records = []
        for model_id, outcome_tag, outcome_col, build_model in [
            ("F3", "FA", "FA_z", fa_mod.build_fa_model),
            ("M2", "MD", "MD_z", md_mod.build_md_model),
        ]:
            spec     = specs[outcome_tag][model_id]
            exposure = spec["exposure"]
            adj_cols = spec.get("adjustment", [])
            for left_out in blocks:
                subset = [b for b in blocks if b != left_out]
                sid    = f"{model_id}_S7_LOBO_{left_out}"
                log.info(f"\n--- {sid} ---")
                data_lobo = build_arrays(rows, exposure, adj_cols, outcome_col,
                                         block_subset=set(subset))
                block_lu = {b: i for i, b in enumerate(subset)}
                sel_rows = [r for r in rows if r["block_id"] in set(subset)
                            and all(r.get(c) is not None
                                    for c in [outcome_col, exposure] + adj_cols)]
                data_lobo["block_idx"] = np.array(
                    [block_lu[r["block_id"]] for r in sel_rows], dtype=int)

                idx_sub = {"roi_types": roi_types, "blocks": subset}
                mdl, _  = build_model(data_lobo, idx_sub)
                be = quick_sample(mdl, sid, result_dir).posterior["beta_exp"].values
                be = be.reshape(-1, len(roi_types))
                for i, roi in enumerate(roi_types):
                    lo, hi = hdi_95(be[:, i])
                    lobo_records.append({
                        "base_model": model_id,
                        "left_out":   left_out,
                        "roi":        roi,
                        "mean":       float(be[:, i].mean()),
                        "sd":         float(be[:, i].std()),
                        "hdi_2.5":    lo,
                        "hdi_97.5":   hi,
                    })
        write_csv(result_dir / "lobo_summary.csv", lobo_records)

    if sens_flags.get("item12_proxy_validity", True):
        run_proxy_validity(result_dir)

    if diag_flags.get("vif_collinearity", True):
        run_vif_collinearity(rows, specs, index_maps, result_dir)
    if diag_flags.get("e_values", True):
        run_evalues(index_maps, result_dir, fa_dir, md_dir)
    if diag_flags.get("imai_rho", True):
        run_imai_rho_sensitivity(index_maps, result_dir, fa_dir, md_dir)
    if diag_flags.get("drop_one_mediator", True):
        run_drop_one_mediator(rows, index_maps, result_dir, fa_dir, md_dir)
    if diag_flags.get("spatial_correlation", True):
        run_spatial_correlation(rows, specs, index_maps, result_dir, fa_dir, md_dir)


def main():
    rows       = load_dataset(DATA_FILE)
    index_maps = json.load(open(IDX_FILE))
    specs      = yaml.safe_load(open(SPEC_FILE))
    run_ctrl   = yaml.safe_load(open(RUN_CTRL))
    sens_flags = run_ctrl["sensitivity_analyses"]
    diag_flags = run_ctrl["diagnostics"]
    scope_cfg  = run_ctrl["analysis_scope"]

    # (tag, global_pooled, include_roi_intercept, analysis_scope key)
    variants = [
        ("B",  False, True,  "per_roi"),
        ("C1", True,  False, "global_c1"),
        ("C2", True,  True,  "global_c2"),
    ]
    for vtag, gp, iri, scope_key in variants:
        if not scope_cfg[scope_key]:
            continue
        results = BASE_DIR / "Results"
        fa_dir  = results / f"FA_{vtag}"
        md_dir  = results / f"MD_{vtag}"
        s_dir   = results / f"Sensitivity_{vtag}"
        s_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"\n>>> variant={vtag}")

        if sens_flags.get("S5_hetero_vs_homo", True):
            run_loo_comparison(rows, specs, index_maps, s_dir, fa_dir,
                               global_pooled=gp, include_roi_intercept=iri)
        if vtag == "B":
            run_regional_analyses(rows, specs, index_maps, sens_flags, diag_flags,
                                  s_dir, fa_dir, md_dir)


if __name__ == "__main__":
    main()
