"""
07_paper_figures.py
===================
Manuscript Figures 4 and 5: regional causal-effect forest plots for FA (F1-F9)
and MD (M1-M8).

Outputs (Figures_paper/):
  FA_regional_forest_combined.png   Figure 4
  MD_regional_forest_combined.png   Figure 5

Usage:
  python 07_paper_figures.py [--dpi 2000] [--out-dir Figures_paper]
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_style import (
    CM_PER_INCH,
    COL,
    FA_MODELS,
    MD_MODELS,
    MODEL_LABEL,
    effect_color,
    load_regional_table,
    ordered_rois,
    rlabel,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "Results"
FIG_WIDTH_CM = 16.0
SUBPLOT_WIDTH_CM = 5.0
SUBPLOT_GAP_CM = 0.5
FIG_HEIGHT_CM = 18.0
PANEL_LEFT = 0.13
PANEL_RIGHT = 0.95


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
            "font.size": 11,
            "axes.titlesize": 9.0,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 2000,
        }
    )


def draw_forest_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    model_id: str,
    active_rois: list[str],
    xlim: tuple[float, float],
    show_xlabel: bool,
    show_ylabels: bool = False,
) -> None:
    sub = df[df["model"] == model_id]
    rows = {
        roi: sub[sub["roi_type"] == roi].iloc[0]
        for roi in active_rois
        if not sub[sub["roi_type"] == roi].empty
    }
    rois = list(rows.keys())
    y = np.arange(len(rois))
    means = np.array([rows[roi]["mean"] for roi in rois], dtype=float)
    lo = np.array([rows[roi]["hdi_2.5"] for roi in rois], dtype=float)
    hi = np.array([rows[roi]["hdi_97.5"] for roi in rois], dtype=float)
    colors = [effect_color(float(rows[roi]["P(>0)"])) for roi in rois]

    ax.errorbar(
        means,
        y,
        xerr=[means - lo, hi - means],
        fmt="none",
        ecolor="#333333",
        elinewidth=0.7,
        capsize=2.2,
        zorder=2,
    )
    ax.scatter(means, y, c=colors, zorder=3, s=16)
    ax.axvline(0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.75, zorder=1)

    ax.set_xlim(*xlim)
    ax.set_ylim(-0.7, len(rois) - 0.3)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels([rlabel(roi) for roi in rois] if show_ylabels else [])
    ax.tick_params(axis="y", length=0, pad=2)

    ax.set_title(MODEL_LABEL.get(model_id, model_id), pad=4)
    if show_xlabel:
        ax.set_xlabel("Standardized Beta (95% HDI)", labelpad=2, fontsize=8.5)


def combined_regional_forest(
    df: pd.DataFrame,
    models: list[str],
    title: str,
    filename: str,
    active_rois: list[str],
    out_dir: Path,
    dpi: int,
) -> Path:
    fig_width_in = FIG_WIDTH_CM / CM_PER_INCH
    fig_height_in = FIG_HEIGHT_CM / CM_PER_INCH
    fig, axes = plt.subplots(3, 3, figsize=(fig_width_in, fig_height_in), squeeze=False)

    panel = df[df["model"].isin(models)]
    x_min = float(panel["hdi_2.5"].min())
    x_max = float(panel["hdi_97.5"].max())
    pad = max((x_max - x_min) * 0.08, 0.05)
    xlim = (x_min - pad, x_max + pad)

    for idx, model_id in enumerate(models):
        row, col = divmod(idx, 3)
        draw_forest_panel(
            axes[row, col],
            df,
            model_id,
            active_rois,
            xlim=xlim,
            show_xlabel=row == 2,
            show_ylabels=col == 0,
        )

    for idx, ax in enumerate(axes.ravel()[len(models):], start=len(models)):
        row, _ = divmod(idx, 3)
        if row == 2:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xlabel("Standardized Beta (95% HDI)", labelpad=2, fontsize=8.5)
        else:
            ax.set_visible(False)

    wspace = SUBPLOT_GAP_CM / SUBPLOT_WIDTH_CM
    fig.subplots_adjust(
        left=PANEL_LEFT,
        right=PANEL_RIGHT,
        top=0.93,
        bottom=0.115,
        wspace=wspace,
        hspace=0.30,
    )
    fig.suptitle(title, fontsize=14, y=0.985)

    handles = [
        mlines.Line2D(
            [],
            [],
            color=COL["pos_credible"],
            marker="o",
            linestyle="None",
            markersize=6,
            label="Credibly positive (pd >= 0.975)",
        ),
        mlines.Line2D(
            [],
            [],
            color=COL["neg_credible"],
            marker="o",
            linestyle="None",
            markersize=6,
            label="Credibly negative (pd >= 0.975)",
        ),
        mlines.Line2D(
            [],
            [],
            color=COL["uncertain"],
            marker="o",
            linestyle="None",
            markersize=6,
            label="Uncertain",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.018), ncol=3)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate manuscript Figures 4 and 5.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "Figures_paper")
    parser.add_argument("--dpi", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()
    active_rois = ordered_rois()
    fa_df = load_regional_table(args.results_dir, "FA")
    md_df = load_regional_table(args.results_dir, "MD")
    output_paths = [
        combined_regional_forest(
            fa_df,
            FA_MODELS,
            "Fractional Anisotropy - Regional Causal Effects",
            "FA_regional_forest_combined.png",
            active_rois,
            args.out_dir,
            args.dpi,
        ),
        combined_regional_forest(
            md_df,
            MD_MODELS,
            "Mean Diffusivity - Regional Causal Effects",
            "MD_regional_forest_combined.png",
            active_rois,
            args.out_dir,
            args.dpi,
        ),
    ]
    for out_path in output_paths:
        log.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
