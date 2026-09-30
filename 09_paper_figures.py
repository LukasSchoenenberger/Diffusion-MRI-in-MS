"""
09_paper_figures.py
===================
Manuscript Figure 3: heatmap of the regional causal effects (posterior mean
standardized beta) for FA and MD.

Output (Figures_paper/): Fig01_regional_causal_effect_heatmap.png

Usage:
  python 09_paper_figures.py [--dpi 2000] [--out-dir Figures_paper]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from paper_style import (
    CM_PER_INCH,
    EXPOSURE_LABEL,
    load_regional_table,
    ordered_rois,
    rlabel,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "Results"

FIG_WIDTH_CM = 16.0
FIG_HEIGHT_CM = 18.0

HEATMAP_CMAP = "RdBu_r"

ESTIMAND_COLUMNS = [
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
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 2000,
        }
    )


def vmax_sym(*arrays: np.ndarray) -> float:
    values = np.concatenate([np.ravel(arr) for arr in arrays])
    vmax = np.nanmax(np.abs(values))
    return float(vmax) if vmax > 0 else 1.0


def build_heatmap_matrix(
    df: pd.DataFrame,
    outcome: str,
    active_rois: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    model_index = 1 if outcome == "FA" else 2
    means = np.full((len(active_rois), len(ESTIMAND_COLUMNS)), np.nan)
    p_gt0 = np.full_like(means, np.nan)

    for col_idx, column in enumerate(ESTIMAND_COLUMNS):
        model_id = column[model_index]
        if model_id is None:
            continue
        sub = df[df["model"] == model_id]
        for row_idx, roi in enumerate(active_rois):
            row = sub[sub["roi_type"] == roi]
            if not row.empty:
                means[row_idx, col_idx] = float(row["mean"].iloc[0])
                p_gt0[row_idx, col_idx] = float(row["P(>0)"].iloc[0])
    return means, p_gt0


def draw_heatmap_panel(
    ax: plt.Axes,
    means: np.ndarray,
    p_gt0: np.ndarray,
    active_rois: list[str],
    subtitle: str,
    norm: TwoSlopeNorm,
    show_xlabels: bool,
) -> plt.AxesImage:
    im = ax.imshow(means, cmap=HEATMAP_CMAP, norm=norm, aspect="auto")

    for row_idx in range(means.shape[0]):
        for col_idx in range(means.shape[1]):
            p = p_gt0[row_idx, col_idx]
            if not np.isnan(p) and 0.025 < p < 0.975:
                ax.add_patch(
                    mpatches.Rectangle(
                        (col_idx - 0.5, row_idx - 0.5),
                        1,
                        1,
                        fill=False,
                        hatch="////",
                        edgecolor="#777777",
                        linewidth=0,
                    )
                )

    ax.set_xticks(range(len(ESTIMAND_COLUMNS)))
    if show_xlabels:
        ax.set_xticklabels(
            [label for label, _, _ in ESTIMAND_COLUMNS],
            rotation=38,
            ha="right",
            fontsize=11,
        )
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

    ax.set_yticks(range(len(active_rois)))
    ax.set_yticklabels([rlabel(roi) for roi in active_rois], fontsize=11)
    ax.set_title(subtitle, fontsize=13, pad=8)
    ax.tick_params(axis="y", labelsize=11)
    return im


def fig_regional_causal_effect_heatmap(
    df_fa: pd.DataFrame,
    df_md: pd.DataFrame,
    active_rois: list[str],
    out_dir: Path,
    dpi: int,
) -> Path:
    fa_means, fa_p = build_heatmap_matrix(df_fa, "FA", active_rois)
    md_means, md_p = build_heatmap_matrix(df_md, "MD", active_rois)
    vmax = vmax_sym(fa_means, md_means)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig_width_in = FIG_WIDTH_CM / CM_PER_INCH
    fig_height_in = FIG_HEIGHT_CM / CM_PER_INCH
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(fig_width_in, fig_height_in),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.18},
    )

    im = draw_heatmap_panel(
        axes[0],
        fa_means,
        fa_p,
        active_rois,
        "Fractional Anisotropy",
        norm,
        show_xlabels=False,
    )
    draw_heatmap_panel(
        axes[1],
        md_means,
        md_p,
        active_rois,
        "Mean Diffusivity",
        norm,
        show_xlabels=True,
    )

    handles = [
        mpatches.Patch(
            facecolor="white",
            edgecolor="#777777",
            hatch="////",
            label="Uncertain direction (pd < 0.975)",
        )
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=1,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("Regional Causal Effects - Heatmap", fontsize=16, y=0.982)
    fig.subplots_adjust(left=0.22, right=0.985, top=0.90, bottom=0.34)

    cbar_ax = fig.add_axes([0.32, 0.13, 0.54, 0.022])
    cb = fig.colorbar(
        im,
        cax=cbar_ax,
        orientation="horizontal",
        label="Standardized Beta",
    )
    cb.ax.tick_params(labelsize=9)
    cb.ax.xaxis.label.set_size(10)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "Fig01_regional_causal_effect_heatmap.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate manuscript Figure 3."
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "Figures_paper")
    parser.add_argument("--dpi", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()
    active_rois = ordered_rois()
    df_fa = load_regional_table(args.results_dir, "FA")
    df_md = load_regional_table(args.results_dir, "MD")
    out_path = fig_regional_causal_effect_heatmap(
        df_fa,
        df_md,
        active_rois,
        args.out_dir,
        args.dpi,
    )
    log.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
