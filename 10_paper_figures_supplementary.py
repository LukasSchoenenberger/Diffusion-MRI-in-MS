"""
10_paper_figures_supplementary.py
=================================
Supplementary Figures S1-S15 (Figures_paper/Supplementary/FigS01-FigS15_*.png).

The pipeline's C1 variant is the G2 analysis (pooled slope, single global
intercept) and C2 is the G1 analysis (pooled slope, ROI-specific intercepts).

Usage:
  python 10_paper_figures_supplementary.py
  python 10_paper_figures_supplementary.py --dpi 800 --only S1 S2
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from paper_style import (
    CM_PER_INCH,
    COL,
    EXPOSURE_SHORT,
    FA_MODELS,
    MD_MODELS,
    MEDIATION_LABEL,
    MODEL_LABEL,
    ROI_COLORS,
    VARIANT_LABEL,
    blabel,
    block_order,
    indexed_rois,
    load_regional_table,
    norm_cols,
    ordered_rois,
    rlabel,
    setup_style,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "Results"

PAGE_WIDTH_CM = 16.0

# Component bar order; "total" is drawn as a reference line.
MEDIATION_COMPONENTS = ["cde", "indirect_myelin", "indirect_coherence", "indirect_cell"]

SENS_MODELS = ["F3", "F6", "F7", "M2", "M5", "M6"]

OBS_COLOR = "#222222"
PP_COLOR = "#0072B2"
PPC_AXIS_NOTE = ("Horizontal axis: standardized outcome; vertical axis: density. "
                 "Axes are scaled independently in each panel.")


def fig_size(width_cm: float, height_cm: float) -> tuple[float, float]:
    return (width_cm / CM_PER_INCH, height_cm / CM_PER_INCH)


def save(fig: plt.Figure, out_dir: Path, name: str, dpi: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    log.info("Saved %s", out_path)
    return out_path


def col_header(model_id: str) -> str:
    return EXPOSURE_SHORT[model_id].replace(" CDE", "\nCDE")


def load_sens(results_dir: Path, variant: str, filename: str) -> pd.DataFrame:
    return norm_cols(pd.read_csv(results_dir / f"Sensitivity_{variant}" / filename))


# ---------------------------------------------------------------------------
# S1-S3  Posterior predictive checks
# ---------------------------------------------------------------------------
def ppc_arrays(path: Path, draw_step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Observed values, thinned posterior-predictive draws and the ROI index."""
    with xr.open_dataset(path, group="observed_data") as obs_ds:
        var = list(obs_ds.data_vars)[0]
        observed = np.asarray(obs_ds[var].values, dtype=float)
    with xr.open_dataset(path, group="posterior_predictive") as pp_ds:
        pp = pp_ds[var].isel(draw=slice(None, None, draw_step)).values
    pp = np.asarray(pp, dtype=float).reshape(-1, pp.shape[-1])
    roi_idx = None
    with xr.open_dataset(path, group="constant_data") as cd_ds:
        if "roi_idx" in cd_ds:
            roi_idx = np.asarray(cd_ds["roi_idx"].values, dtype=int)
    return observed, pp, roi_idx


def draw_ppc_panel(ax: plt.Axes, observed: np.ndarray, pp: np.ndarray) -> None:
    lo, hi = np.percentile(observed, [0.5, 99.5])
    if not np.isfinite(lo) or hi <= lo:
        ax.set_visible(False)
        return
    bins = np.linspace(lo, hi, 41)
    centers = 0.5 * (bins[:-1] + bins[1:])
    obs_density, _ = np.histogram(observed, bins=bins, density=True)
    pp_density = np.stack([np.histogram(row, bins=bins, density=True)[0] for row in pp])
    band_lo, band_hi = np.percentile(pp_density, [2.5, 97.5], axis=0)

    ax.fill_between(centers, band_lo, band_hi, color=PP_COLOR, alpha=0.30, linewidth=0)
    ax.plot(centers, np.median(pp_density, axis=0), color=PP_COLOR, linewidth=0.6)
    ax.plot(centers, obs_density, color=OBS_COLOR, linewidth=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def ppc_legend_handles() -> list:
    return [
        mlines.Line2D([], [], color=OBS_COLOR, linewidth=1.2, label="Observed"),
        mpatches.Patch(
            facecolor=PP_COLOR,
            alpha=0.30,
            edgecolor=PP_COLOR,
            label="Posterior predictive, 95% interval and median",
        ),
    ]


def fig_ppc_regional(
    results_dir: Path,
    out_dir: Path,
    outcome: str,
    name: str,
    dpi: int,
    draw_step: int,
) -> Path:
    models = FA_MODELS if outcome == "FA" else MD_MODELS
    rois = ordered_rois()
    all_rois = indexed_rois()

    fig, axes = plt.subplots(
        len(rois),
        len(models),
        figsize=fig_size(PAGE_WIDTH_CM, 20.0),
        squeeze=False,
    )
    for col, model_id in enumerate(models):
        observed, pp, roi_idx = ppc_arrays(results_dir / f"{outcome}_B" / f"{model_id}_idata.nc", draw_step)
        for row, roi in enumerate(rois):
            ax = axes[row][col]
            mask = roi_idx == all_rois.index(roi)
            if mask.sum() < 5:
                ax.set_visible(False)
                continue
            draw_ppc_panel(ax, observed[mask], pp[:, mask])
            if row == 0:
                ax.set_title(col_header(model_id), fontsize=7.0, pad=4)
            if col == 0:
                ax.set_ylabel(rlabel(roi), fontsize=7.5, rotation=0, ha="right", va="center", labelpad=4)
        del observed, pp

    fig.suptitle(
        f"Posterior predictive checks - {'fractional anisotropy' if outcome == 'FA' else 'mean diffusivity'}, "
        "regional analysis",
        fontsize=11,
        y=0.985,
    )
    fig.text(0.5, 0.038, PPC_AXIS_NOTE, ha="center", fontsize=7, color="#555555")
    fig.subplots_adjust(left=0.11, right=0.99, top=0.935, bottom=0.070, wspace=0.12, hspace=0.30)
    fig.legend(handles=ppc_legend_handles(), loc="lower center", bbox_to_anchor=(0.5, 0.002), ncol=2, fontsize=8)
    return save(fig, out_dir, name, dpi)


def fig_ppc_global(results_dir: Path, out_dir: Path, dpi: int, draw_step: int) -> Path:
    # Columns follow the FA estimand order; MD has no coherence model, so that
    # column stays empty rather than shifting every MD panel one place left.
    column_models = {
        "FA": FA_MODELS,
        "MD": ["M1", None, "M2", "M3", "M4", "M5", "M6", "M7", "M8"],
    }
    # (variant tag, outcome) in manuscript reading order: G1 before G2.
    blocks = [("C2", "FA"), ("C1", "FA"), ("C2", "MD"), ("C1", "MD")]
    ncols = len(FA_MODELS)

    fig, axes = plt.subplots(len(blocks), ncols, figsize=fig_size(PAGE_WIDTH_CM, 12.0), squeeze=False)
    for row, (variant, outcome) in enumerate(blocks):
        for col, model_id in enumerate(column_models[outcome]):
            ax = axes[row][col]
            if row == 0:
                ax.set_title(col_header(FA_MODELS[col]), fontsize=7.0, pad=4)
            if model_id is None:
                ax.set_visible(False)
                continue
            observed, pp, _ = ppc_arrays(
                results_dir / f"{outcome}_{variant}" / f"{model_id}_idata.nc", draw_step
            )
            draw_ppc_panel(ax, observed, pp)
            if col == 0:
                ax.set_ylabel(
                    f"{VARIANT_LABEL[variant].replace(' analysis', '')}\n{outcome}",
                    fontsize=7.5,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=4,
                )
            del observed, pp

    fig.suptitle("Posterior predictive checks - G1 and G2 analyses", fontsize=11, y=0.975)
    fig.text(0.5, 0.085, PPC_AXIS_NOTE, ha="center", fontsize=7, color="#555555")
    fig.subplots_adjust(left=0.10, right=0.99, top=0.86, bottom=0.14, wspace=0.12, hspace=0.42)
    fig.legend(handles=ppc_legend_handles(), loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2, fontsize=8)
    return save(fig, out_dir, "FigS03_ppc_global_variants.png", dpi)


# ---------------------------------------------------------------------------
# S4  PSIS-LOO
# ---------------------------------------------------------------------------
def fig_psis_loo(results_dir: Path, out_dir: Path, dpi: int) -> Path:
    panels = [("B", "Regional analysis"), ("C1", VARIANT_LABEL["C1"]), ("C2", VARIANT_LABEL["C2"])]
    specs = [("heteroscedastic_B", "Heteroscedastic", COL["primary"], -0.16),
             ("homoscedastic_B", "Homoscedastic", COL["uncertain"], 0.16)]

    fig, axes = plt.subplots(1, 3, figsize=fig_size(PAGE_WIDTH_CM, 6.4), squeeze=False, sharey=True)
    axes = axes.ravel()
    for ax, (variant, title) in zip(axes, panels):
        df = load_sens(results_dir, variant, "tier2_looic.csv")
        models = sorted(df["model"].unique())
        y = np.arange(len(models))
        for key, label, color, offset in specs:
            sub = df[df["variant"] == key].set_index("model")
            means = np.array([sub.loc[m, "elpd_loo"] for m in models], dtype=float)
            errs = np.array([sub.loc[m, "se"] for m in models], dtype=float)
            ax.errorbar(means, y + offset, xerr=errs, fmt="none", ecolor="#333333",
                        elinewidth=0.7, capsize=2.0, zorder=2)
            ax.scatter(means, y + offset, color=color, s=18, zorder=3, label=label)
        het = df[df["variant"] == "heteroscedastic_B"].set_index("model")["elpd_loo"]
        hom = df[df["variant"] == "homoscedastic_B"].set_index("model")["elpd_loo"]
        for i, m in enumerate(models):
            ax.annotate(f"$\\Delta$ELPD {het[m] - hom[m]:+,.0f}", (het[m], y[i] - 0.42),
                        fontsize=6.5, ha="right", va="center", color="#333333")
        ax.set_yticks(y)
        ax.set_yticklabels([EXPOSURE_SHORT[m] for m in models])
        ax.set_ylim(-0.7, len(models) - 0.3)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9.5, pad=6)
        ax.set_xlabel("ELPD (PSIS-LOO)", fontsize=8.5)
        ax.tick_params(labelsize=8)

    handles = [mlines.Line2D([], [], color=c, marker="o", linestyle="None", markersize=5, label=l)
               for _, l, c, _ in specs]
    fig.suptitle("Residual-variance specification compared by PSIS-LOO", fontsize=11, y=0.98)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.78, bottom=0.30, wspace=0.16)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=8.5)
    return save(fig, out_dir, "FigS04_psis_loo.png", dpi)


# ---------------------------------------------------------------------------
# S5  Block random effects
# ---------------------------------------------------------------------------
def read_summary(results_dir: Path, outcome: str, model_id: str) -> pd.DataFrame:
    return norm_cols(pd.read_csv(results_dir / f"{outcome}_B" / f"{model_id}_summary.csv"))


def fig_block_random_effects(results_dir: Path, out_dir: Path, dpi: int) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=fig_size(PAGE_WIDTH_CM, 11.0), squeeze=False)

    for col, (outcome, models, ref_model) in enumerate(
        [("FA", FA_MODELS, "F1"), ("MD", MD_MODELS, "M1")]
    ):
        rows = [read_summary(results_dir, outcome, m).set_index("parameter") for m in models]
        means = np.array([r.loc["sigma_block", "mean"] for r in rows], dtype=float)
        lo = np.array([r.loc["sigma_block", "hdi_2.5"] for r in rows], dtype=float)
        hi = np.array([r.loc["sigma_block", "hdi_97.5"] for r in rows], dtype=float)
        x = np.arange(len(models))

        ax = axes[0][col]
        ax.bar(x, means, color=COL["primary"], alpha=0.78, width=0.62)
        ax.errorbar(x, means, yerr=[means - lo, hi - means], fmt="none",
                    ecolor="#333333", elinewidth=0.7, capsize=2.2)
        ax.set_xticks(x)
        ax.set_xticklabels([EXPOSURE_SHORT[m] for m in models], fontsize=6.5, rotation=45, ha="right")
        ax.set_ylabel(r"$\sigma_{\mathrm{block}}$ (SD units)", fontsize=8.5)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{'Fractional anisotropy' if outcome == 'FA' else 'Mean diffusivity'}",
                     fontsize=9.5, pad=6)
        ax.tick_params(axis="y", labelsize=8)

        summary = read_summary(results_dir, outcome, ref_model).set_index("parameter")
        blocks = summary.loc[[f"v_block[{b}]" for b in block_order()]].copy()
        blocks["label"] = [blabel(b) for b in block_order()]
        xb = np.arange(len(blocks))
        ax = axes[1][col]
        ax.errorbar(xb, blocks["mean"].to_numpy(float),
                    yerr=[blocks["mean"].to_numpy(float) - blocks["hdi_2.5"].to_numpy(float),
                          blocks["hdi_97.5"].to_numpy(float) - blocks["mean"].to_numpy(float)],
                    fmt="none", ecolor="#333333", elinewidth=0.7, capsize=2.2)
        ax.scatter(xb, blocks["mean"].to_numpy(float), color=COL["primary"], s=16, zorder=3)
        ax.axhline(0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.75)
        ax.set_xticks(xb)
        ax.set_xticklabels(blocks["label"], fontsize=6.5, rotation=40, ha="right")
        ax.set_ylabel("Block offset (SD units)", fontsize=8.5)
        ax.set_title(f"Per-block intercepts, {MODEL_LABEL[ref_model]}", fontsize=9.0, pad=6)
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle("Between-block variance", fontsize=11, y=0.985)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.90, bottom=0.09, wspace=0.30, hspace=0.72)
    return save(fig, out_dir, "FigS05_block_random_effects.png", dpi)


# ---------------------------------------------------------------------------
# S6/S7  Axon mediation decomposition, all ROIs
# ---------------------------------------------------------------------------
def fig_mediation_all_rois(results_dir: Path, out_dir: Path, outcome: str, name: str, dpi: int) -> Path:
    model = "F10" if outcome == "FA" else "M9"
    df = norm_cols(pd.read_csv(results_dir / f"{outcome}_B" / f"{model}_gcomputation_by_roi_summary.csv"))
    rois = ordered_rois()
    present = set(df["component"])
    comps = [c for c in MEDIATION_COMPONENTS if c in present]

    panel = df[df["component"].isin(comps) & df["roi"].isin(rois)]
    ymin = float(panel["hdi_2.5"].min())
    ymax = float(panel["hdi_97.5"].max())
    pad = (ymax - ymin) * 0.10
    ylim = (ymin - pad, ymax + pad)

    fig, axes = plt.subplots(2, 5, figsize=fig_size(PAGE_WIDTH_CM, 11.0), squeeze=False, sharey=True)
    for idx, roi in enumerate(rois):
        ax = axes[idx // 5][idx % 5]
        sub = df[df["roi"] == roi].set_index("component")
        x = np.arange(len(comps))
        means = np.array([sub.loc[c, "mean"] for c in comps], dtype=float)
        lo = np.array([sub.loc[c, "hdi_2.5"] for c in comps], dtype=float)
        hi = np.array([sub.loc[c, "hdi_97.5"] for c in comps], dtype=float)
        ax.bar(x, means, color=[COL[c] for c in comps], alpha=0.85, width=0.68)
        ax.errorbar(x, means, yerr=[means - lo, hi - means], fmt="none",
                    ecolor="#333333", elinewidth=0.6, capsize=1.8)
        ax.axhline(0, color="#333333", linewidth=0.7)
        if "total" in set(sub.index):
            total = float(sub.loc["total", "mean"])
            ax.axhline(total, color=COL["total"], linewidth=0.9, linestyle="--")
            ax.annotate(f"total {total:+.2f}", (len(comps) - 0.45, total), fontsize=6.0,
                        ha="right", va="bottom", color=COL["total"])
        ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.set_title(rlabel(roi), fontsize=8.5, pad=4)
        if idx % 5 == 0:
            ax.set_ylabel(r"Standardized $\beta$", fontsize=8.5)
        ax.tick_params(axis="y", labelsize=7.5)

    handles = [mpatches.Patch(color=COL[c], alpha=0.85, label=MEDIATION_LABEL[c]) for c in comps]
    handles.append(mlines.Line2D([], [], color=COL["total"], linestyle="--", linewidth=1.0,
                                 label="Total effect"))
    label = "fractional anisotropy" if outcome == "FA" else "mean diffusivity"
    fig.suptitle(f"Axon density mediation decomposition by ROI - {label}", fontsize=11, y=0.985)
    fig.subplots_adjust(left=0.105, right=0.99, top=0.885, bottom=0.16, wspace=0.16, hspace=0.28)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=len(handles), fontsize=7.5)
    return save(fig, out_dir, name, dpi)


# ---------------------------------------------------------------------------
# S8  Pooled / global mediation decomposition
# ---------------------------------------------------------------------------
def fig_mediation_pooled(results_dir: Path, out_dir: Path, dpi: int) -> Path:
    variants = [("B", "Regional analysis, pooled", COL["primary"]),
                ("C2", VARIANT_LABEL["C2"], "#009E73"),
                ("C1", VARIANT_LABEL["C1"], "#E69F00")]

    fig, axes = plt.subplots(1, 2, figsize=fig_size(PAGE_WIDTH_CM, 7.4), squeeze=False)
    axes = axes.ravel()
    for ax, outcome in zip(axes, ["FA", "MD"]):
        model = "F10" if outcome == "FA" else "M9"
        tables = {
            tag: norm_cols(pd.read_csv(results_dir / f"{outcome}_{tag}" / f"{model}_gcomputation_summary.csv"))
            .set_index("component")
            for tag, _, _ in variants
        }
        comps = [c for c in ["total"] + MEDIATION_COMPONENTS
                 if any(c in t.index for t in tables.values())]
        x = np.arange(len(comps))
        width = 0.26
        for i, (tag, label, color) in enumerate(variants):
            table = tables[tag]
            offset = (i - 1) * width
            means, lo, hi = [], [], []
            for c in comps:
                if c in table.index:
                    means.append(float(table.loc[c, "mean"]))
                    lo.append(float(table.loc[c, "hdi_2.5"]))
                    hi.append(float(table.loc[c, "hdi_97.5"]))
                else:
                    means.append(np.nan)
                    lo.append(np.nan)
                    hi.append(np.nan)
            means = np.array(means)
            ax.bar(x + offset, means, width=width, color=color, alpha=0.85, label=label)
            ax.errorbar(x + offset, means,
                        yerr=[means - np.array(lo), np.array(hi) - means],
                        fmt="none", ecolor="#333333", elinewidth=0.6, capsize=1.8)
        ax.axhline(0, color="#333333", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([MEDIATION_LABEL[c] for c in comps], fontsize=7.5, rotation=22, ha="right")
        ax.set_ylabel(r"Standardized $\beta$", fontsize=8.5)
        ax.set_title("Fractional anisotropy" if outcome == "FA" else "Mean diffusivity",
                     fontsize=9.5, pad=6)
        ax.tick_params(axis="y", labelsize=8)

    handles = [mpatches.Patch(color=c, alpha=0.85, label=l) for _, l, c in variants]
    fig.suptitle("Compartment-spanning axon density mediation decomposition", fontsize=11, y=0.985)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.845, bottom=0.315, wspace=0.26)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=3, fontsize=8)
    return save(fig, out_dir, "FigS08_axon_mediation_pooled.png", dpi)


# ---------------------------------------------------------------------------
# S9/S10/S11  Tier-1 sensitivity forests
# ---------------------------------------------------------------------------
def fig_tier1_sensitivity(
    results_dir: Path,
    out_dir: Path,
    variants: list[tuple[str, str, str]],
    title: str,
    name: str,
    dpi: int,
) -> Path:
    tier1 = load_sens(results_dir, "B", "tier1_summary.csv")
    primary = {
        "FA": load_regional_table(results_dir, "FA"),
        "MD": load_regional_table(results_dir, "MD"),
    }
    rois = ordered_rois()
    series = [("__primary__", "Primary model, prior N(0, 1)", COL["primary"])] + variants
    offsets = np.linspace(-0.26, 0.26, len(series))

    fig, axes = plt.subplots(2, 3, figsize=fig_size(PAGE_WIDTH_CM, 15.0), squeeze=False)
    for idx, model_id in enumerate(SENS_MODELS):
        ax = axes[idx // 3][idx % 3]
        outcome = "FA" if model_id.startswith("F") else "MD"
        y = np.arange(len(rois))
        for (key, _, color), offset in zip(series, offsets):
            if key == "__primary__":
                src = primary[outcome]
                sub = src[src["model"] == model_id].set_index("roi_type")
            else:
                src = tier1[(tier1["model"] == model_id) & (tier1["sensitivity"] == key)]
                sub = src.set_index("roi")
            available = [r for r in rois if r in sub.index]
            if not available:
                continue
            pos = np.array([rois.index(r) for r in available], dtype=float) + offset
            means = sub.loc[available, "mean"].to_numpy(float)
            lo = sub.loc[available, "hdi_2.5"].to_numpy(float)
            hi = sub.loc[available, "hdi_97.5"].to_numpy(float)
            ax.errorbar(means, pos, xerr=[means - lo, hi - means], fmt="none",
                        ecolor="#333333", elinewidth=0.6, capsize=1.6, zorder=2)
            ax.scatter(means, pos, color=color, s=14, zorder=3)
        ax.axvline(0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.75)
        ax.set_ylim(-0.7, len(rois) - 0.3)
        ax.invert_yaxis()
        ax.set_yticks(y)
        ax.set_yticklabels([rlabel(r) for r in rois] if idx % 3 == 0 else [])
        ax.tick_params(axis="y", length=0, labelsize=8)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_title(MODEL_LABEL[model_id], fontsize=8.5, pad=4)
        if idx // 3 == 1:
            ax.set_xlabel(r"Standardized $\beta$ (95% HDI)", fontsize=8.5)

    handles = [mlines.Line2D([], [], color=color, marker="o", linestyle="None", markersize=5, label=label)
               for _, label, color in series]
    fig.suptitle(title, fontsize=11, y=0.985)
    fig.subplots_adjust(left=0.135, right=0.985, top=0.925, bottom=0.115, wspace=0.14, hspace=0.20)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.012),
               ncol=min(3, len(series)), fontsize=8)
    return save(fig, out_dir, name, dpi)


# ---------------------------------------------------------------------------
# S12  Leave-one-block-out
# ---------------------------------------------------------------------------
def fig_lobo(results_dir: Path, out_dir: Path, dpi: int) -> Path:
    lobo = load_sens(results_dir, "B", "lobo_summary.csv")
    primary = {"FA": load_regional_table(results_dir, "FA"), "MD": load_regional_table(results_dir, "MD")}
    rois = ordered_rois()
    pairs = [("F3", "FA"), ("M2", "MD")]

    fig, axes = plt.subplots(1, 2, figsize=fig_size(PAGE_WIDTH_CM, 7.6), squeeze=False)
    axes = axes.ravel()
    for ax, (model_id, outcome) in zip(axes, pairs):
        sub = lobo[lobo["base_model"] == model_id]
        present = set(sub["left_out"].unique())
        blocks = [b for b in block_order() if b in present]
        x = np.arange(len(blocks))
        ref = primary[outcome]
        ref = ref[ref["model"] == model_id].set_index("roi_type")
        for roi in rois:
            roi_rows = sub[sub["roi"] == roi].set_index("left_out")
            if roi_rows.empty:
                continue
            values = np.array([roi_rows.loc[b, "mean"] if b in roi_rows.index else np.nan
                               for b in blocks], dtype=float)
            ax.plot(x, values, marker="o", markersize=2.6, linewidth=0.9,
                    color=ROI_COLORS[roi], label=rlabel(roi))
            if roi in ref.index:
                ax.axhline(float(ref.loc[roi, "mean"]), color=ROI_COLORS[roi],
                           linewidth=0.6, linestyle=":", alpha=0.8)
        ax.axhline(0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([blabel(b) for b in blocks], fontsize=7, rotation=45, ha="right")
        ax.set_xlabel("Block left out", fontsize=8.5)
        ax.set_ylabel(r"Standardized $\beta$", fontsize=8.5)
        ax.set_title(MODEL_LABEL[model_id], fontsize=9.5, pad=6)
        ax.tick_params(axis="y", labelsize=8)

    handles = [mlines.Line2D([], [], color=ROI_COLORS[r], marker="o", markersize=4,
                             linewidth=1.0, label=rlabel(r)) for r in rois]
    fig.suptitle("Leave-one-block-out stability of the myelin density effect", fontsize=11, y=0.985)
    fig.subplots_adjust(left=0.095, right=0.99, top=0.845, bottom=0.315, wspace=0.24)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=5, fontsize=7.5)
    return save(fig, out_dir, "FigS12_leave_one_block_out.png", dpi)


# ---------------------------------------------------------------------------
# S13  Drop-one-mediator
# ---------------------------------------------------------------------------
DROP_LABEL = {
    "__full__": "Full model",
    "myelin_density": "− Myelin Density",
    "axon_coherence": "− Fiber Coherence",
    "cell_density": "− Cell Density",
}
DROP_COLOR = {
    "__full__": COL["primary"],
    "myelin_density": COL["indirect_myelin"],
    "axon_coherence": COL["indirect_coherence"],
    "cell_density": COL["indirect_cell"],
}


def fig_drop_one_mediator(results_dir: Path, out_dir: Path, dpi: int) -> Path:
    drops = load_sens(results_dir, "B", "drop_one_mediator.csv")
    rois = ordered_rois()
    specs = [("FA", "F10"), ("MD", "M9")]

    fig, axes = plt.subplots(2, 2, figsize=fig_size(PAGE_WIDTH_CM, 14.5), squeeze=False)
    used_conditions: list[str] = []
    for col, (outcome, gmodel) in enumerate(specs):
        full = norm_cols(
            pd.read_csv(results_dir / f"{outcome}_B" / f"{gmodel}_gcomputation_by_roi_summary.csv")
        )
        sub_all = drops[drops["gcomp_model"] == gmodel]
        conditions = ["__full__"] + sorted(sub_all["dropped_mediator"].unique())
        used_conditions = sorted(set(used_conditions) | set(conditions), key=list(DROP_LABEL).index)
        offsets = np.linspace(-0.28, 0.28, len(conditions))

        for row, component in enumerate(["cde", "total"]):
            ax = axes[row][col]
            for condition, offset in zip(conditions, offsets):
                if condition == "__full__":
                    src = full[full["component"] == component].set_index("roi")
                else:
                    src = sub_all[
                        (sub_all["dropped_mediator"] == condition)
                        & (sub_all["component"] == component)
                    ].set_index("roi")
                available = [r for r in rois if r in src.index]
                if not available:
                    continue
                pos = np.array([rois.index(r) for r in available], dtype=float) + offset
                means = src.loc[available, "mean"].to_numpy(float)
                lo = src.loc[available, "hdi_2.5"].to_numpy(float)
                hi = src.loc[available, "hdi_97.5"].to_numpy(float)
                ax.errorbar(means, pos, xerr=[means - lo, hi - means], fmt="none",
                            ecolor="#333333", elinewidth=0.5, capsize=1.4, zorder=2)
                ax.scatter(means, pos, color=DROP_COLOR[condition], s=13, zorder=3)
            ax.axvline(0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.75)
            ax.set_ylim(-0.7, len(rois) - 0.3)
            ax.invert_yaxis()
            ax.set_yticks(np.arange(len(rois)))
            ax.set_yticklabels([rlabel(r) for r in rois] if col == 0 else [])
            ax.tick_params(axis="y", length=0, labelsize=8)
            ax.tick_params(axis="x", labelsize=7.5)
            outcome_name = "Fractional anisotropy" if outcome == "FA" else "Mean diffusivity"
            ax.set_title(f"{outcome_name} - {'CDE' if component == 'cde' else 'total effect'}",
                         fontsize=9.0, pad=4)
            if row == 1:
                ax.set_xlabel(r"Standardized $\beta$ (95% HDI)", fontsize=8.5)

    handles = [mlines.Line2D([], [], color=DROP_COLOR[c], marker="o", linestyle="None",
                             markersize=5, label=DROP_LABEL[c]) for c in used_conditions]
    fig.suptitle("Drop-one-mediator diagnostic for the axon density decomposition", fontsize=11, y=0.985)
    fig.subplots_adjust(left=0.135, right=0.985, top=0.925, bottom=0.135, wspace=0.10, hspace=0.34)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.006), ncol=4, fontsize=8)
    return save(fig, out_dir, "FigS13_drop_one_mediator.png", dpi)


# ---------------------------------------------------------------------------
# S14  Imai rho mediation sensitivity
# ---------------------------------------------------------------------------
IMAI_LABEL = {
    "gcomp_indirect_myelin": "Indirect via Myelin Density",
    "gcomp_indirect_coherence": "Indirect via Fiber Coherence",
    "gcomp_indirect_cell": "Indirect via Cell Density",
    "total_indirect_F1-F7": "Total indirect effect",
    "total_indirect_M1-M6": "Total indirect effect",
}


def fig_imai_rho(results_dir: Path, out_dir: Path, dpi: int) -> Path:
    df = load_sens(results_dir, "B", "imai_rho_sensitivity.csv")
    rois = ordered_rois()
    df = df[df["roi"].isin(rois)]
    layout = {
        "FA": ["gcomp_indirect_myelin", "gcomp_indirect_coherence", "gcomp_indirect_cell",
               "total_indirect_F1-F7"],
        "MD": ["gcomp_indirect_myelin", "gcomp_indirect_cell", "total_indirect_M1-M6"],
    }

    fig = plt.figure(figsize=fig_size(PAGE_WIDTH_CM, 17.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.15])
    markers = ["o", "^", "s", "D"]

    for row, outcome in enumerate(["FA", "MD"]):
        comps = layout[outcome]
        for col in range(4):
            if col >= len(comps):
                continue
            component = comps[col]
            ax = fig.add_subplot(gs[row, col])
            sub = df[(df["outcome"] == outcome) & (df["component"] == component)]
            at_zero = sub[np.isclose(sub["rho"], 0.0)]["indirect_obs"]
            scale = float(np.nanmax(np.abs(at_zero))) if len(at_zero) else 0.0
            limit = max(3.0 * scale, 0.05)
            for roi in rois:
                curve = sub[sub["roi"] == roi].sort_values("rho")
                if curve.empty:
                    continue
                ax.plot(curve["rho"], curve["indirect_adj"], linewidth=0.8, color=ROI_COLORS[roi])
            ax.axhline(0, color="#444444", linewidth=0.7, linestyle="--", alpha=0.7)
            ax.set_ylim(-limit, limit)
            ax.set_title(f"{outcome} - {IMAI_LABEL[component]}", fontsize=7.5, pad=4)
            ax.set_xlabel(r"$\rho$", fontsize=8)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel(r"Adjusted indirect $\beta$", fontsize=8)

    for col, outcome in enumerate(["FA", "MD"]):
        ax = fig.add_subplot(gs[2, col * 2:(col + 1) * 2])
        comps = layout[outcome]
        for i, component in enumerate(comps):
            sub = df[(df["outcome"] == outcome) & (df["component"] == component)]
            crit = sub.groupby("roi")["rho_crit"].first()
            available = [r for r in rois if r in crit.index]
            if not available:
                continue
            ax.scatter(crit.loc[available].to_numpy(float),
                       [rois.index(r) for r in available],
                       marker=markers[i % len(markers)], s=20,
                       facecolors="none", edgecolors=[ROI_COLORS[r] for r in available],
                       linewidths=0.9, zorder=3)
        ax.axvspan(-0.2, 0.2, color="#cccccc", alpha=0.45, zorder=0)
        for v in (-0.5, 0.0, 0.5):
            ax.axvline(v, color="#888888", linewidth=0.6, linestyle=":", zorder=1)
        ax.set_ylim(-0.7, len(rois) - 0.3)
        ax.invert_yaxis()
        ax.set_yticks(np.arange(len(rois)))
        ax.set_yticklabels([rlabel(r) for r in rois] if col == 0 else [])
        ax.tick_params(axis="y", length=0, labelsize=7.5)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_xlabel(r"$\rho_{\mathrm{crit}}$", fontsize=8)
        ax.set_title(f"Critical $\\rho$ - {'fractional anisotropy' if outcome == 'FA' else 'mean diffusivity'}",
                     fontsize=8.5, pad=4)

    marker_handles = [
        mlines.Line2D([], [], color="#555555", marker=markers[i % len(markers)], linestyle="None",
                      markersize=5, markerfacecolor="none",
                      label=IMAI_LABEL[c].replace("Indirect via ", "via ").replace(" Density", ""))
        for i, c in enumerate(layout["FA"])
    ]
    marker_handles.append(mpatches.Patch(facecolor="#cccccc", alpha=0.45,
                                         label=r"Fragile, $|\rho|<0.2$"))
    roi_handles = [mlines.Line2D([], [], color=ROI_COLORS[r], linewidth=1.4, label=rlabel(r)) for r in rois]
    fig.suptitle("Imai sensitivity of the mediation decomposition to mediator-outcome confounding",
                 fontsize=11, y=0.985)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.930, bottom=0.145, wspace=0.34, hspace=0.60)
    fig.legend(handles=marker_handles, loc="lower center", bbox_to_anchor=(0.5, 0.062), ncol=5, fontsize=7.0)
    fig.legend(handles=roi_handles, loc="lower center", bbox_to_anchor=(0.5, 0.004), ncol=5, fontsize=7.5)
    return save(fig, out_dir, "FigS14_imai_rho.png", dpi)


# ---------------------------------------------------------------------------
# S15  Proxy-validity simulation
# ---------------------------------------------------------------------------
def fig_proxy_validity(results_dir: Path, out_dir: Path, dpi: int) -> Path:
    df = norm_cols(pd.read_csv(results_dir / "Sensitivity_B" / "item12_proxy_validity_summary.csv"))
    proxy_rois = ["WM", "WM_partial_demyel", "Lesion_core"]
    panels = [
        ("proxy", "gamma_om", r"Proxy quality $\gamma_{\mathrm{om}}$",
         r"Grid 1: proxy quality (effect scale $=1.0$)", None),
        ("effect", "beta_axon_scale", "Effect-size scale",
         r"Grid 2: effect size ($\gamma_{\mathrm{om}}=0.6$)", 1.0),
    ]

    fig, axes = plt.subplots(1, 2, figsize=fig_size(PAGE_WIDTH_CM, 7.2), squeeze=False)
    axes = axes.ravel()
    for ax, (grid, xcol, xlabel, title, marker_x) in zip(axes, panels):
        sub = df[df["grid"] == grid]
        for roi in proxy_rois:
            rows = sub[sub["roi"] == roi].sort_values(xcol)
            if rows.empty:
                continue
            x = rows[xcol].to_numpy(float)
            true_beta = rows["true_beta_z"].to_numpy(float)
            ax.fill_between(x,
                            rows["hdi_2.5"].to_numpy(float) - true_beta,
                            rows["hdi_97.5"].to_numpy(float) - true_beta,
                            color=ROI_COLORS[roi], alpha=0.15, linewidth=0)
            ax.plot(x, rows["bias"].to_numpy(float), marker="o", markersize=3.2,
                    linewidth=1.0, color=ROI_COLORS[roi], label=rlabel(roi))
        ax.axhline(0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.75)
        if marker_x is not None:
            ax.axvline(marker_x, color="#888888", linewidth=0.7, linestyle=":")
            ax.annotate("observed effect size", (marker_x, ax.get_ylim()[1]), fontsize=6.5,
                        rotation=90, ha="right", va="top", color="#666666")
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_ylabel("Bias in the axon CDE (SD units)", fontsize=8.5)
        ax.set_title(title, fontsize=9.0, pad=6)
        ax.tick_params(labelsize=8)

    handles = [mlines.Line2D([], [], color=ROI_COLORS[r], marker="o", markersize=4.5,
                             linewidth=1.2, label=rlabel(r)) for r in proxy_rois]
    handles.append(mpatches.Patch(facecolor="#999999", alpha=0.30,
                                  label="95% HDI, centred on the true effect"))
    fig.suptitle("Proxy-adjustment validity for the latent oligodendrocyte confounder",
                 fontsize=11, y=0.985)
    fig.subplots_adjust(left=0.105, right=0.99, top=0.835, bottom=0.30, wspace=0.28)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=4, fontsize=7.5)
    return save(fig, out_dir, "FigS15_proxy_validity.png", dpi)


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the supplementary paper figures.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "Figures_paper" / "Supplementary")
    parser.add_argument("--dpi", type=int, default=2000)
    parser.add_argument("--draw-step", type=int, default=20,
                        help="Posterior-predictive draw thinning for S1-S3.")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Subset of figure keys, e.g. --only S1 S12")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()
    rd, od, dpi = args.results_dir, args.out_dir, args.dpi

    builders = {
        "S1": lambda: fig_ppc_regional(rd, od, "FA", "FigS01_ppc_FA_regional.png", dpi, args.draw_step),
        "S2": lambda: fig_ppc_regional(rd, od, "MD", "FigS02_ppc_MD_regional.png", dpi, args.draw_step),
        "S3": lambda: fig_ppc_global(rd, od, dpi, args.draw_step),
        "S4": lambda: fig_psis_loo(rd, od, dpi),
        "S5": lambda: fig_block_random_effects(rd, od, dpi),
        "S6": lambda: fig_mediation_all_rois(rd, od, "FA", "FigS06_axon_mediation_all_rois_FA.png", dpi),
        "S7": lambda: fig_mediation_all_rois(rd, od, "MD", "FigS07_axon_mediation_all_rois_MD.png", dpi),
        "S8": lambda: fig_mediation_pooled(rd, od, dpi),
        "S9": lambda: fig_tier1_sensitivity(
            rd, od,
            [("S1_StudentT", "Student-$t$ likelihood ($\\nu=4$)", "#D55E00")],
            "Sensitivity to the likelihood: Student-$t$ versus the primary specification",
            "FigS09_sensitivity_student_t.png", dpi),
        "S10": lambda: fig_tier1_sensitivity(
            rd, od,
            [("S2a_prior2.5", "Prior N(0, 2.5)", "#E69F00"),
             ("S2b_prior5.0", "Prior N(0, 5)", "#009E73")],
            "Sensitivity to the prior scale on the fixed effects",
            "FigS10_sensitivity_prior_scale.png", dpi),
        "S11": lambda: fig_tier1_sensitivity(
            rd, od,
            [("S4_noPatchWeight", "No patch-size weighting ($\\delta_n=0$)", "#CC79A7")],
            "Sensitivity to the patch-size weighting of the residual scale",
            "FigS11_sensitivity_patch_weighting.png", dpi),
        "S12": lambda: fig_lobo(rd, od, dpi),
        "S13": lambda: fig_drop_one_mediator(rd, od, dpi),
        "S14": lambda: fig_imai_rho(rd, od, dpi),
        "S15": lambda: fig_proxy_validity(rd, od, dpi),
    }

    keys = args.only if args.only else list(builders)
    for key in keys:
        if key not in builders:
            raise SystemExit(f"Unknown figure key: {key}")
        builders[key]()


if __name__ == "__main__":
    main()
