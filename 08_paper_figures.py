"""
08_paper_figures.py
===================
Manuscript Figures 6, 7 and 8.

Outputs (Figures_paper/):
  Fig05_cde_vs_total.png                              Figure 6, total effect vs CDE
  Fig04_axon_mediation_lesion_core.png                Figure 7, axon mediation in LC
  Fig10_variant_comparison_roi_level_confounding.png  Figure 8, G1 vs G2 analysis

Usage:
  python 08_paper_figures.py [--dpi 2000] [--out-dir Figures_paper]
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_style import (
    CM_PER_INCH,
    COL,
    EXPOSURE_LABEL,
    MEDIATION_LABEL as COMPONENT_LABEL,
    VARIANT_LABEL,
    load_estimand_table,
    load_gcomp_table,
    load_regional_table,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "Results"

FIG_WIDTH_CM = 16.0
FIG_HEIGHT_CM = FIG_WIDTH_CM * 5.5 / 12.0

# Manuscript Table 4 order; cortical layers 1-3 enter as one pooled point.
FIG05_ROIS = [
    ("WM", "NAWM", "#0072B2"),
    ("Surrounding_WM", "SWM", "#56B4E9"),
    ("WM_partial_demyel", "PDWM", "#009E73"),
    ("Perilesion", "PL", "#E69F00"),
    ("Cortex", "Cortex I-III (pooled)", "#CC79A7"),
    ("Cortex_diffuse_demyel", "DDC", "#D55E00"),
    ("Cortex_partial_demyel", "PDC", "#882255"),
]
CORTEX_POOL = ["Cortex_1", "Cortex_2", "Cortex_3"]

# Bar order in the mediation panels; "total" is drawn as a reference line instead.
MEDIATION_COMPONENTS = ["cde", "indirect_myelin", "indirect_coherence", "indirect_cell"]


VARIANT_ROWS = [
    (EXPOSURE_LABEL["F1"], "F1", "M1"),
    (EXPOSURE_LABEL["F2"], "F2", None),
    (EXPOSURE_LABEL["F3"], "F3", "M2"),
    (EXPOSURE_LABEL["F4"], "F4", "M3"),
    (EXPOSURE_LABEL["F5"], "F5", "M4"),
    (EXPOSURE_LABEL["F6"], "F6", "M5"),
    (EXPOSURE_LABEL["F7"], "F7", "M6"),
    (EXPOSURE_LABEL["F8"], "F8", "M7"),
    (EXPOSURE_LABEL["F9"], "F9", "M8"),
]
# C1 = pooled slope + single global intercept  -> manuscript G2 analysis
# C2 = pooled slope + ROI-specific intercepts  -> manuscript G1 analysis
VARIANT_SPECS = [
    ("C1", VARIANT_LABEL["C1"], "#E69F00", -0.12, 3),
    ("C2", VARIANT_LABEL["C2"], "#009E73", 0.12, 5),
]


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
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 2000,
        }
    )


def get_point(df: pd.DataFrame, model_id: str, roi_key: str) -> float:
    sub = df[df["model"] == model_id]
    if roi_key == "Cortex":
        rows = sub[sub["roi_type"].isin(CORTEX_POOL)]
        if rows.empty:
            return np.nan
        return float(rows["mean"].mean())
    row = sub[sub["roi_type"] == roi_key]
    if row.empty:
        return np.nan
    return float(row["mean"].iloc[0])


def draw_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    pairs: list[tuple[str, str, str]],
    subtitle: str,
    markers: list[str],
) -> list[mlines.Line2D]:
    shape_handles = []
    for (total_id, cde_id, predictor_name), marker in zip(pairs, markers):
        for roi_key, _, color in FIG05_ROIS:
            total_effect = get_point(df, total_id, roi_key)
            cde = get_point(df, cde_id, roi_key)
            if np.isnan(total_effect) or np.isnan(cde):
                continue
            ax.scatter(total_effect, cde, color=color, marker=marker, s=32, alpha=0.88, zorder=5)

    for (_, _, predictor_name), marker in zip(pairs, markers):
        shape_handles.append(
            mlines.Line2D(
                [],
                [],
                color="gray",
                marker=marker,
                markersize=5.0,
                linestyle="None",
                label=predictor_name,
            )
        )

    ax.axline((0, 0), slope=1, color="#444444", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.axvline(0, color="#aaaaaa", linewidth=0.5, linestyle=":")
    ax.axhline(0, color="#aaaaaa", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Total Effect - Standardized β", fontsize=8)
    ax.set_ylabel("Controlled Direct Effect - Standardized β", fontsize=8)
    ax.set_title(subtitle, fontsize=10.5, pad=8)
    ax.tick_params(axis="both", labelsize=8.5)
    return shape_handles


def fig_cde_vs_total(df_fa: pd.DataFrame, df_md: pd.DataFrame, out_dir: Path, dpi: int) -> Path:
    pairs_fa = [
        ("F1", "F7", EXPOSURE_LABEL["F1"]),
        ("F4", "F8", EXPOSURE_LABEL["F4"]),
        ("F5", "F9", EXPOSURE_LABEL["F5"]),
    ]
    pairs_md = [
        ("M1", "M6", EXPOSURE_LABEL["M1"]),
        ("M3", "M7", EXPOSURE_LABEL["M3"]),
        ("M4", "M8", EXPOSURE_LABEL["M4"]),
    ]
    markers = ["o", "s", "^"]

    fig_width_in = FIG_WIDTH_CM / CM_PER_INCH
    fig_height_in = FIG_HEIGHT_CM / CM_PER_INCH
    fig, axes = plt.subplots(1, 2, figsize=(fig_width_in, fig_height_in), squeeze=False)
    axes = axes.ravel()

    shape_handles = draw_panel(axes[0], df_fa, pairs_fa, "Fractional Anisotropy", markers)
    draw_panel(axes[1], df_md, pairs_md, "Mean Diffusivity", markers)
    axes[1].xaxis.set_major_locator(mticker.MultipleLocator(0.1))

    roi_handles = [mpatches.Patch(color=color, label=label) for _, label, color in FIG05_ROIS]
    diag_line = mlines.Line2D(
        [],
        [],
        color="#444444",
        linewidth=1.2,
        linestyle="--",
        label="Total = CDE (no mediation)",
    )
    handles = roi_handles + shape_handles + [diag_line]

    fig.suptitle("Total Effect vs. Controlled Direct Effect", fontsize=15, y=0.985)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.80, bottom=0.36, wspace=0.32)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=4, fontsize=7.5)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "Fig05_cde_vs_total.png"
    fig.savefig(out_path, dpi=max(dpi, 2000))
    plt.close(fig)
    return out_path


def draw_lesion_core_mediation_panel(ax: plt.Axes, df: pd.DataFrame, subtitle: str) -> None:
    roi_df = df[df["roi"] == "Lesion_core"]
    present = set(roi_df["component"])
    comp_order = [c for c in MEDIATION_COMPONENTS if c in present]
    x = np.arange(len(comp_order))
    vals = []
    lo = []
    hi = []
    colors = []
    for component in comp_order:
        row = roi_df[roi_df["component"] == component]
        vals.append(float(row["mean"].iloc[0]) if not row.empty else 0.0)
        lo.append(float(row["hdi_2.5"].iloc[0]) if not row.empty else 0.0)
        hi.append(float(row["hdi_97.5"].iloc[0]) if not row.empty else 0.0)
        colors.append(COL.get(component, "#888888"))

    vals = np.asarray(vals)
    ax.bar(x, vals, color=colors, alpha=0.84, width=0.62)
    ax.errorbar(x, vals, yerr=[vals - np.asarray(lo), np.asarray(hi) - vals], fmt="none",
                ecolor="#333333", elinewidth=0.7, capsize=3)
    ax.axhline(0, color="#333333", linewidth=0.8)

    total_row = roi_df[roi_df["component"] == "total"]
    if not total_row.empty:
        total = float(total_row["mean"].iloc[0])
        ax.axhline(total, color=COL["total"], linewidth=1.0, linestyle="--")
        ax.text(
            len(comp_order) - 0.55,
            total,
            f"Total {total:+.3f}",
            fontsize=7.5,
            va="bottom",
            ha="right",
            color=COL["total"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([COMPONENT_LABEL[c] for c in comp_order], rotation=28, ha="right", fontsize=8.5)
    ax.set_ylabel("Standardized β (95% HDI)", fontsize=9)
    ax.set_title(subtitle, fontsize=10.5, pad=8)
    ax.tick_params(axis="y", labelsize=8.5)


def fig_lesion_core_mediation(df_fa: pd.DataFrame, df_md: pd.DataFrame, out_dir: Path, dpi: int) -> Path:
    fig_width_in = FIG_WIDTH_CM / CM_PER_INCH
    fig_height_in = FIG_HEIGHT_CM / CM_PER_INCH
    fig, axes = plt.subplots(1, 2, figsize=(fig_width_in, fig_height_in), squeeze=False)
    axes = axes.ravel()

    draw_lesion_core_mediation_panel(axes[0], df_fa, "Fractional Anisotropy")
    draw_lesion_core_mediation_panel(axes[1], df_md, "Mean Diffusivity")

    fig.suptitle("Axon Density Mediation Decomposition - Lesion Core (LC)", fontsize=15, y=0.985)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.80, bottom=0.28, wspace=0.32)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "Fig04_axon_mediation_lesion_core.png"
    fig.savefig(out_path, dpi=max(dpi, 2000))
    plt.close(fig)
    return out_path


def get_variant_point(df: pd.DataFrame, model_id: str | None) -> tuple[float, float, float] | None:
    if model_id is None:
        return None
    row = df[df["model"] == model_id]
    if row.empty:
        return None
    return (
        float(row["mean"].iloc[0]),
        float(row["hdi_2.5"].iloc[0]),
        float(row["hdi_97.5"].iloc[0]),
    )


def draw_variant_panel(
    ax: plt.Axes,
    tables: dict[str, pd.DataFrame],
    outcome: str,
    subtitle: str,
    xlim: tuple[float, float],
    show_ylabels: bool,
) -> None:
    row_gap = 1.18
    y = np.arange(len(VARIANT_ROWS)) * row_gap
    outcome_index = 1 if outcome == "FA" else 2

    for variant, label, color, offset, zorder in VARIANT_SPECS:
        df = tables[variant]
        for idx, row_spec in enumerate(VARIANT_ROWS):
            model_id = row_spec[outcome_index]
            point = get_variant_point(df, model_id)
            if point is None:
                # No such estimand for this outcome (MD has no coherence model).
                if model_id is None:
                    ax.text(0, y[idx], "-", ha="center", va="center", fontsize=10.5, color="#555555")
                continue
            mean, lo, hi = point
            ax.errorbar(
                mean,
                y[idx] + offset,
                xerr=[[mean - lo], [hi - mean]],
                fmt="none",
                ecolor="#333333",
                elinewidth=0.7,
                capsize=2.2,
                capthick=0.7,
                alpha=0.95,
                zorder=zorder,
            )
            ax.scatter(mean, y[idx] + offset, color=color, zorder=zorder + 0.1, s=16)

    ax.axvline(0, color="#444444", linewidth=0.85, linestyle="--", alpha=0.75)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.7 * row_gap, y[-1] + 0.7 * row_gap)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels([row[0] for row in VARIANT_ROWS] if show_ylabels else [])
    ax.tick_params(axis="y", labelsize=8.5, length=0, pad=2)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.set_xlabel("Standardized β", fontsize=8.5)
    ax.set_title(subtitle, fontsize=10.5, pad=8)


def collect_variant_xlim(tables_by_outcome: dict[str, dict[str, pd.DataFrame]]) -> tuple[float, float]:
    values = []
    for outcome, tables in tables_by_outcome.items():
        outcome_index = 1 if outcome == "FA" else 2
        for variant in ["C1", "C2"]:
            for row_spec in VARIANT_ROWS:
                point = get_variant_point(tables[variant], row_spec[outcome_index])
                if point is not None:
                    _, lo, hi = point
                    values.extend([lo, hi])
    x_min = min(values)
    x_max = max(values)
    pad = max((x_max - x_min) * 0.08, 0.04)
    return x_min - pad, x_max + pad


def fig_variant_comparison_refined(results_dir: Path, out_dir: Path, dpi: int) -> Path:
    tables_by_outcome = {
        "FA": {
            "C1": load_estimand_table(results_dir, "FA", "C1"),
            "C2": load_estimand_table(results_dir, "FA", "C2"),
        },
        "MD": {
            "C1": load_estimand_table(results_dir, "MD", "C1"),
            "C2": load_estimand_table(results_dir, "MD", "C2"),
        },
    }
    xlim = collect_variant_xlim(tables_by_outcome)

    fig_width_in = FIG_WIDTH_CM / CM_PER_INCH
    fig_height_in = FIG_HEIGHT_CM / CM_PER_INCH
    fig, axes = plt.subplots(1, 2, figsize=(fig_width_in, fig_height_in), squeeze=False)
    axes = axes.ravel()

    draw_variant_panel(
        axes[0],
        tables_by_outcome["FA"],
        "FA",
        "Fractional Anisotropy",
        xlim,
        show_ylabels=True,
    )
    draw_variant_panel(
        axes[1],
        tables_by_outcome["MD"],
        "MD",
        "Mean Diffusivity",
        xlim,
        show_ylabels=False,
    )

    # Legend lists G1 before G2.
    handles = [
        mlines.Line2D([], [], color=color, marker="o", markersize=6, linestyle="-", label=label)
        for _, label, color, _, _ in sorted(VARIANT_SPECS, key=lambda spec: spec[1])
    ]
    fig.suptitle("ROI-Level Confounding", fontsize=15, y=0.985)
    fig.subplots_adjust(left=0.2, right=0.985, top=0.80, bottom=0.32, wspace=0.16)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=9.5)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "Fig10_variant_comparison_roi_level_confounding.png"
    fig.savefig(out_path, dpi=max(dpi, 2000))
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate manuscript Figures 6-8.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "Figures_paper")
    parser.add_argument("--dpi", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()
    df_fa = load_regional_table(args.results_dir, "FA")
    df_md = load_regional_table(args.results_dir, "MD")
    gcomp_fa = load_gcomp_table(args.results_dir, "FA")
    gcomp_md = load_gcomp_table(args.results_dir, "MD")
    output_paths = [
        fig_cde_vs_total(df_fa, df_md, args.out_dir, args.dpi),
        fig_lesion_core_mediation(gcomp_fa, gcomp_md, args.out_dir, args.dpi),
        fig_variant_comparison_refined(args.results_dir, args.out_dir, args.dpi),
    ]
    for out_path in output_paths:
        log.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
