"""
02_FA_models.py
===============
Bayesian hierarchical regression of fractional anisotropy (FA), one model per
estimand (config/model_specs.yaml).

F1-F9, for subpatch i in ROI type r and block b:
    FA_i        ~ Normal(mu_i, sigma_i)
    mu_i        = alpha_r + beta_exp_r * x_i + sum_p beta_adj_{r,p} * z_{p,i}
                  + gamma' tech_i + v_b,             v_b ~ Normal(0, sigma_block)
    log sigma_i = delta0 + delta_r + delta_n * log(n_i / n_ref)

F10 decomposes the axon effect into the controlled direct effect (F7) and
interventional indirect effects via myelin density, fiber coherence and cell
density by parametric Monte Carlo g-computation.

Variants (config/run_control.yaml, analysis_scope):
    B   regional analysis: ROI-specific intercepts and slopes   Results/FA_B
    C1  G2 analysis: pooled slopes, single global intercept     Results/FA_C1
    C2  G1 analysis: pooled slopes, ROI-specific intercepts     Results/FA_C2

Outputs per variant directory:
    F{n}_idata.nc, F{n}_summary.csv, F{n}_diagnostics.txt
    F10_mediator_{mediator}_idata.nc, F10_gcomputation_summary.csv
    F10_gcomputation_by_roi_summary.csv, posterior_corr_myelin_cell_FA.csv  (B only)
"""

import csv
import json
import logging
import math
from pathlib import Path

import numpy as np
import pymc as pm
import arviz as az
import xarray as xr
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent
DATA_FILE = BASE_DIR / "Analysis_Dataset" / "analysis_dataset.csv"
IDX_FILE  = BASE_DIR / "Analysis_Dataset" / "index_maps.json"
SPEC_FILE = BASE_DIR / "config" / "model_specs.yaml"
RUN_CTRL  = BASE_DIR / "config" / "run_control.yaml"

OUTCOME_COL = "FA_z"
TECH_COLS   = ("FixDur_days_z", "StorTime_days_z")

N_DRAWS       = 2000
N_TUNE        = 1000
N_CHAINS      = 4
TARGET_ACCEPT = 0.9
RANDOM_SEED   = 42

N_GCOMP_DRAWS = 500

# Mediator models: Axon + Tech + block RE; the cell mediator additionally
# adjusts for GFAP and microglia.
MEDIATOR_ADJ_COLS = {"cell_density": ["gfap_density", "microglia_density"]}


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


def build_arrays(rows, exposure_col, adj_cols, outcome_col=OUTCOME_COL):
    """Complete-case design arrays for one model."""
    required = [outcome_col, exposure_col] + adj_cols
    sel = [r for r in rows if all(r.get(c) is not None for c in required)]
    n_dropped = len(rows) - len(sel)
    if n_dropped:
        log.info(f"  Complete-case: dropped {n_dropped}/{len(rows)} rows with missing values")

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


def build_fa_model(data, index_maps, *, global_pooled=False, include_roi_intercept=True):
    """
    B   -> global_pooled=False, include_roi_intercept=True
    C1  -> global_pooled=True,  include_roi_intercept=False
    C2  -> global_pooled=True,  include_roi_intercept=True
    delta_r stays ROI-indexed and the technical covariates stay pooled in all variants.
    """
    n_adj  = data["X_adj"].shape[1]
    coords = {"roi": index_maps["roi_types"], "block": index_maps["blocks"]}
    if n_adj > 0:
        coords["adj"] = data["adj_cols"]

    with pm.Model(coords=coords) as model:
        sigma_block = pm.HalfNormal("sigma_block", sigma=1.0)
        v_block     = pm.Normal("v_block", mu=0, sigma=sigma_block, dims="block")

        if include_roi_intercept:
            alpha      = pm.Normal("alpha", mu=0, sigma=1.0, dims="roi")
            alpha_term = alpha[data["roi_idx"]]
        else:
            alpha0     = pm.Normal("alpha0", mu=0, sigma=1.0)
            alpha_term = alpha0

        if global_pooled:
            beta_exp      = pm.Normal("beta_exp", mu=0, sigma=1.0)
            beta_exp_term = beta_exp * data["X_exp"]
        else:
            beta_exp      = pm.Normal("beta_exp", mu=0, sigma=1.0, dims="roi")
            beta_exp_term = beta_exp[data["roi_idx"]] * data["X_exp"]

        if n_adj > 0:
            if global_pooled:
                beta_adj   = pm.Normal("beta_adj", mu=0, sigma=1.0, dims="adj")
                adj_effect = pm.math.dot(data["X_adj"], beta_adj)
            else:
                beta_adj   = pm.Normal("beta_adj", mu=0, sigma=1.0, dims=("roi", "adj"))
                adj_effect = (beta_adj[data["roi_idx"]] * data["X_adj"]).sum(axis=-1)
        else:
            adj_effect = 0.0

        gamma       = pm.Normal("gamma", mu=0, sigma=1.0, shape=len(TECH_COLS))
        tech_effect = pm.math.dot(data["X_tech"], gamma)

        delta0        = pm.Normal("delta0", mu=0, sigma=1.0)
        sigma_delta_r = pm.HalfNormal("sigma_delta_r", sigma=0.5)
        delta_r       = pm.Normal("delta_r", mu=0, sigma=sigma_delta_r, dims="roi")
        delta_n       = pm.Normal("delta_n", mu=-0.25, sigma=0.5)
        log_sigma = delta0 + delta_r[data["roi_idx"]] + delta_n * data["log_n"]
        sigma_eff = pm.Deterministic("sigma_eff", pm.math.exp(log_sigma))

        mu = alpha_term + beta_exp_term + adj_effect + tech_effect + v_block[data["block_idx"]]

        pm.Normal("FA_obs", mu=mu, sigma=sigma_eff, observed=data["y"])

    return model, coords


def hdi_95(draws):
    lo, hi = az.hdi(np.asarray(draws, dtype=float), hdi_prob=0.95)
    return float(lo), float(hi)


def write_csv(path, records):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    log.info(f"  Saved {path}")


def sample_and_save(model, model_id, result_dir, data):
    nc_path = result_dir / f"{model_id}_idata.nc"
    if nc_path.exists():
        log.info(f"  {model_id}: loading existing {nc_path.name}")
        return az.from_netcdf(str(nc_path))

    log.info(f"  Sampling {model_id} ...")
    with model:
        idata = pm.sample(
            draws=N_DRAWS,
            tune=N_TUNE,
            chains=N_CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            idata_kwargs={"log_likelihood": True},
            progressbar=True,
        )
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)

    idata.add_groups({"constant_data": xr.Dataset(
        {"roi_idx": xr.DataArray(data["roi_idx"], dims=["obs_id"])}
    )})
    idata.to_netcdf(str(nc_path))
    log.info(f"  Saved {nc_path}")

    rhat = az.rhat(idata)
    ess  = az.ess(idata)
    divs = int(idata.sample_stats.diverging.values.sum())
    diag_txt = [
        f"Model: {model_id}",
        f"Divergences: {divs}",
        f"R-hat max: {float(max(rhat[v].values.max() for v in rhat.data_vars)):.4f}",
        f"ESS bulk min: {float(min(ess[v].values.min() for v in ess.data_vars)):.1f}",
        "",
        "R-hat per parameter group:",
    ]
    for v in sorted(rhat.data_vars):
        vals = rhat[v].values.ravel()
        diag_txt.append(f"  {v}: max={vals.max():.4f}  mean={vals.mean():.4f}")
    (result_dir / f"{model_id}_diagnostics.txt").write_text("\n".join(diag_txt))

    summary = az.summary(idata, var_names=[v for v in idata.posterior.data_vars
                                           if v != "sigma_eff"], hdi_prob=0.95)
    write_csv(result_dir / f"{model_id}_summary.csv",
              [{"parameter": p, **summary.loc[p].to_dict()} for p in summary.index])
    return idata


def fit_mediator_model(rows, mediator_col, index_maps, result_dir, *,
                       global_pooled=False, include_roi_intercept=True):
    """Axon -> mediator model (same structure as the FA models)."""
    nc_path = result_dir / f"F10_mediator_{mediator_col}_idata.nc"
    if nc_path.exists():
        log.info(f"  mediator_{mediator_col}: loading existing {nc_path.name}")
        return az.from_netcdf(str(nc_path))

    data = build_arrays(rows, "axon_density", MEDIATOR_ADJ_COLS.get(mediator_col, []),
                        outcome_col=mediator_col + "_z")
    model, _ = build_fa_model(data, index_maps, global_pooled=global_pooled,
                              include_roi_intercept=include_roi_intercept)
    with model:
        idata = pm.sample(draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS,
                          target_accept=TARGET_ACCEPT, random_seed=RANDOM_SEED,
                          progressbar=True)
    idata.to_netcdf(str(nc_path))
    return idata


def gcomputation_F10(rows, index_maps, cde_idata, mediator_idatas, result_dir,
                     outcome_col=OUTCOME_COL, *, global_pooled=False,
                     include_roi_intercept=True):
    """Parametric Monte Carlo g-computation for Axon -> FA mediation.

    For each of N_GCOMP_DRAWS posterior draws s and the contrast x = +0.5 vs
    x' = -0.5 SD (dx = 1):
      1. Sample M_k(x)_i and M_k(x')_i from each mediator model (Normal
         residuals with the mediator's heteroscedastic sigma_i).
      2. With the F7 slopes:
           cde_s        = mean_i[beta_exp_i * dx]
           indirect_k_s = mean_i[beta_adj_{i,k} * (M_k(x)_i - M_k(x')_i)]
           total_s      = cde_s + sum_k indirect_k_s
    Intercepts, block effects and technical covariates cancel in every
    contrast. Mediators missing from `mediator_idatas` contribute zero
    (used by the drop-one-mediator diagnostic in 04_sensitivity_analysis.py).
    Components are summarized over the draws by mean, SD, 95% HDI and P(>0);
    per-ROI summaries are written for the regional analysis only.
    """
    log.info(f"  Running F10 g-computation (global_pooled={global_pooled}) ...")
    rng = np.random.default_rng(RANDOM_SEED)

    cde_post  = cde_idata.posterior
    n_samples = cde_post.sizes["chain"] * cde_post.sizes["draw"]
    roi_types = index_maps["roi_types"]
    n_roi     = len(roi_types)

    adj_cols_f7 = ["myelin_density", "axon_coherence", "cell_density",
                   "gfap_density", "microglia_density"]
    data_f7 = build_arrays(rows, "axon_density", adj_cols_f7, outcome_col=outcome_col)
    roi_idx = data_f7["roi_idx"]
    log_n   = data_f7["log_n"]
    N       = len(roi_idx)

    mediator_cols = ["myelin_density", "axon_coherence", "cell_density"]

    def _flat(post, var):
        v = post[var].values
        return v.reshape((-1,) + v.shape[2:])

    F7_be = _flat(cde_post, "beta_exp")   # B: (S, R)     C: (S,)
    F7_ba = _flat(cde_post, "beta_adj")   # B: (S, R, A)  C: (S, A)

    med_params = {}
    for col, idata in mediator_idatas.items():
        mp = idata.posterior
        med_params[col] = {
            "beta_exp": _flat(mp, "beta_exp"),
            "delta0":   _flat(mp, "delta0"),
            "delta_r":  _flat(mp, "delta_r"),
            "delta_n":  _flat(mp, "delta_n"),
        }

    def _beta_exp_out_at(s):
        if F7_be.ndim == 2:
            return F7_be[s][roi_idx]
        return np.full(N, float(F7_be[s]))

    def _axon_on_med_at(col, s):
        be = med_params[col]["beta_exp"]
        if be.ndim == 2:
            return be[s][roi_idx]
        return np.full(N, float(be[s]))

    def _med_on_y_at(col, s):
        a = adj_cols_f7.index(col)
        if F7_ba.ndim == 3:
            return F7_ba[s][roi_idx, a]
        return np.full(N, float(F7_ba[s, a]))

    def _med_sigma_at(col, s):
        p = med_params[col]
        return np.exp(float(p["delta0"][s]) + p["delta_r"][s][roi_idx]
                      + float(p["delta_n"][s]) * log_n)

    comp_keys = ["total", "cde", "indirect_myelin", "indirect_coherence", "indirect_cell"]
    results   = {k: [] for k in comp_keys}
    save_per_roi = (not global_pooled) and include_roi_intercept
    if save_per_roi:
        roi_results = {r: {k: [] for k in comp_keys} for r in range(n_roi)}

    draw_idx = rng.integers(0, n_samples, size=N_GCOMP_DRAWS)
    x, xprime = 0.5, -0.5
    dx = x - xprime

    for s in draw_idx:
        be_out = _beta_exp_out_at(s)

        med_delta = {}   # M_k(x)_i - M_k(x')_i
        ba_s      = {}
        for col in mediator_cols:
            if col not in med_params:
                med_delta[col] = np.zeros(N)
                ba_s[col]      = np.zeros(N)
                continue
            sig    = _med_sigma_at(col, s)
            eps_x  = rng.normal(0.0, sig)
            eps_xp = rng.normal(0.0, sig)
            med_delta[col] = _axon_on_med_at(col, s) * dx + (eps_x - eps_xp)
            ba_s[col]      = _med_on_y_at(col, s)

        cde_i   = be_out * dx
        ind_i   = {c: ba_s[c] * med_delta[c] for c in mediator_cols}
        total_i = cde_i + sum(ind_i.values())

        results["cde"].append(float(cde_i.mean()))
        results["total"].append(float(total_i.mean()))
        results["indirect_myelin"].append(float(ind_i["myelin_density"].mean()))
        results["indirect_coherence"].append(float(ind_i["axon_coherence"].mean()))
        results["indirect_cell"].append(float(ind_i["cell_density"].mean()))

        if save_per_roi:
            for r in range(n_roi):
                mask = roi_idx == r
                roi_results[r]["cde"].append(float(cde_i[mask].mean()))
                roi_results[r]["total"].append(float(total_i[mask].mean()))
                roi_results[r]["indirect_myelin"].append(float(ind_i["myelin_density"][mask].mean()))
                roi_results[r]["indirect_coherence"].append(float(ind_i["axon_coherence"][mask].mean()))
                roi_results[r]["indirect_cell"].append(float(ind_i["cell_density"][mask].mean()))

    summary_rows = []
    for key, vals in results.items():
        arr = np.array(vals)
        lo, hi = hdi_95(arr)
        summary_rows.append({
            "component": key,
            "mean":      float(arr.mean()),
            "sd":        float(arr.std()),
            "hdi_2.5%":  lo,
            "hdi_97.5%": hi,
            "P(>0)":     float((arr > 0).mean()),
        })
    write_csv(result_dir / "F10_gcomputation_summary.csv", summary_rows)

    if save_per_roi:
        roi_summary_rows = []
        for r, roi_name in enumerate(roi_types):
            for key, vals in roi_results[r].items():
                arr = np.asarray(vals, dtype=float)
                lo, hi = hdi_95(arr)
                roi_summary_rows.append({
                    "roi":       roi_name,
                    "component": key,
                    "mean":      float(arr.mean()),
                    "sd":        float(arr.std()),
                    "hdi_2.5%":  lo,
                    "hdi_97.5%": hi,
                    "P(>0)":     float((arr > 0).mean()),
                })
        write_csv(result_dir / "F10_gcomputation_by_roi_summary.csv", roi_summary_rows)

    return results


def run_fa_variant(rows, index_maps, specs, fa_flags, result_dir, *,
                   global_pooled=False, include_roi_intercept=True):
    variant_kw = dict(global_pooled=global_pooled, include_roi_intercept=include_roi_intercept)

    fitted_idatas = {}
    for model_id, spec in specs.items():
        if spec["type"] != "regression" or not fa_flags.get(model_id, True):
            continue
        log.info(f"\n--- {model_id}: {spec['label']} ---")
        adj_cols = spec.get("adjustment", [])
        data = build_arrays(rows, spec["exposure"], adj_cols)
        log.info(f"  N={data['n']}, exposure={spec['exposure']}, adj={adj_cols}")
        model, _ = build_fa_model(data, index_maps, **variant_kw)
        fitted_idatas[model_id] = sample_and_save(model, model_id, result_dir, data)

    if fa_flags.get("F10", True):
        log.info("\n--- F10: Axon -> FA mediation (g-computation) ---")
        if "F7" not in fitted_idatas:
            data_f7 = build_arrays(rows, specs["F7"]["exposure"], specs["F7"]["adjustment"])
            model_f7, _ = build_fa_model(data_f7, index_maps, **variant_kw)
            fitted_idatas["F7"] = sample_and_save(model_f7, "F7", result_dir, data_f7)

        mediator_idatas = {
            med_col: fit_mediator_model(rows, med_col, index_maps, result_dir, **variant_kw)
            for med_col in ["myelin_density", "axon_coherence", "cell_density"]
        }
        gcomputation_F10(rows, index_maps, fitted_idatas["F7"], mediator_idatas,
                         result_dir, **variant_kw)

    # Posterior correlation of the myelin and cell slopes in the axon CDE model
    nc_f7 = result_dir / "F7_idata.nc"
    if not global_pooled and nc_f7.exists():
        roi_types = index_maps["roi_types"]
        adj_cols  = specs["F7"]["adjustment"]
        ba = az.from_netcdf(str(nc_f7)).posterior["beta_adj"].values
        ba = ba.reshape(-1, len(roi_types), len(adj_cols))
        my_idx = adj_cols.index("myelin_density")
        ce_idx = adj_cols.index("cell_density")
        write_csv(result_dir / "posterior_corr_myelin_cell_FA.csv", [
            {"model": "F7", "roi": roi,
             "corr_myelin_cell": float(np.corrcoef(ba[:, r, my_idx], ba[:, r, ce_idx])[0, 1])}
            for r, roi in enumerate(roi_types)
        ])


def main():
    rows       = load_dataset(DATA_FILE)
    index_maps = json.load(open(IDX_FILE))
    specs      = yaml.safe_load(open(SPEC_FILE))["FA"]
    run_ctrl   = yaml.safe_load(open(RUN_CTRL))
    fa_flags   = run_ctrl["FA_models"]
    scope_cfg  = run_ctrl["analysis_scope"]
    log.info(f"Dataset: {len(rows)} subpatches")

    # (directory, global_pooled, include_roi_intercept, analysis_scope key)
    variants = [
        ("FA_B",  False, True,  "per_roi"),
        ("FA_C1", True,  False, "global_c1"),
        ("FA_C2", True,  True,  "global_c2"),
    ]
    for dir_tag, gp, iri, scope_key in variants:
        if not scope_cfg[scope_key]:
            continue
        result_dir = BASE_DIR / "Results" / dir_tag
        result_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"\n>>> variant={dir_tag}")
        run_fa_variant(rows, index_maps, specs, fa_flags, result_dir,
                       global_pooled=gp, include_roi_intercept=iri)


if __name__ == "__main__":
    main()
