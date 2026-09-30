# Statistical analysis code

Code for the causal Bayesian analysis in *A mesoscopic causal descriptor of tissue
composition for interpreting diffusion MRI in multiple sclerosis* (Schönenberger et al.).

Subpatch-level FA and MD from 9.4 T ex vivo diffusion MRI are modelled as functions of
co-registered histology-derived densities (axon, myelin, cell, astrocyte/GFAP,
microglia, fiber coherence) with DAG-derived adjustment sets, in hierarchical Bayesian
regressions fitted with PyMC. The prespecified statistical analysis plan is
`Statistical-Analysis-Plan.pdf`.

## Requirements

```bash
conda env create -f environment.yml
conda activate diffusion-causal
```

## Input data

The subpatch extraction CSVs are not part of this repository (see the manuscript's
Data and code availability statement). `01_prepare_data.py` expects them in
`Data_Extraction/`, one file per block and metric, named
`block_<block id>_<metric>.csv` with `<metric>` one of `FA`, `MD`, `axon_density`,
`axon_coherence`, `cell_density`, `gfap_density`, `microglia_density`,
`myelin_density`. Each file has one row per subpatch with the columns `sub_patch`
(`<ROI name>_<index>`), `metric_artifact_mask` (subpatch mean) and
`metric_artifact_mask_n_voxels` (number of valid voxels).

## Pipeline

Run from this directory, in order:

| Script | Purpose | Output |
|---|---|---|
| `01_prepare_data.py` | ROI pooling, technical covariates, normalization | `Analysis_Dataset/` |
| `02_FA_models.py` | FA models F1-F9, mediation F10 | `Results/FA_{B,C1,C2}/` |
| `03_MD_models.py` | MD models M1-M8, mediation M9 | `Results/MD_{B,C1,C2}/` |
| `04_sensitivity_analysis.py` | Sensitivity analyses and diagnostics | `Results/Sensitivity_{B,C1,C2}/` |
| `05_reporting.py` | G2 - G1 slope differences | `Results/Table_C1_vs_C2_gap.csv` |
| `07_paper_figures.py` | Figures 4, 5 | `Figures_paper/` |
| `08_paper_figures.py` | Figures 6, 7, 8 | `Figures_paper/` |
| `09_paper_figures.py` | Figure 3 | `Figures_paper/` |
| `10_paper_figures_supplementary.py` | Figures S1-S15 | `Figures_paper/Supplementary/` |
| `11_table4.py` | Table 4 | `Table4_regional_effects.{docx,md}` |

`config/run_control.yaml` selects the models and analyses to run; all analyses in
the manuscript are enabled. Models whose `*_idata.nc` already exists are loaded
instead of refitted. Estimands and adjustment sets are defined in
`config/model_specs.yaml` (manuscript Table 2).

Analysis variants and their manuscript names:

| Tag | Manuscript | Slopes | Intercepts |
|---|---|---|---|
| `B`  | Regional analysis | ROI-specific | ROI-specific |
| `C1` | G2 analysis | pooled | single global |
| `C2` | G1 analysis | pooled | ROI-specific |

## Figures

| Manuscript | File | Script |
|---|---|---|
| Figure 3 | `Fig01_regional_causal_effect_heatmap.png` | `09_paper_figures.py` |
| Figure 4 | `FA_regional_forest_combined.png` | `07_paper_figures.py` |
| Figure 5 | `MD_regional_forest_combined.png` | `07_paper_figures.py` |
| Figure 6 | `Fig05_cde_vs_total.png` | `08_paper_figures.py` |
| Figure 7 | `Fig04_axon_mediation_lesion_core.png` | `08_paper_figures.py` |
| Figure 8 | `Fig10_variant_comparison_roi_level_confounding.png` | `08_paper_figures.py` |
| Figures S1-S15 | `Supplementary/FigS01_*.png` ... `FigS15_*.png` | `10_paper_figures_supplementary.py` |

## Tables

| Manuscript | Source |
|---|---|
| Table 3, Table S1 | `Analysis_Dataset/sample_size_table.csv` |
| Table 4 | `11_table4.py` |
| Tables S2, S3 | Regional posteriors `Results/{FA,MD}_B/*_idata.nc`, summarized with `paper_style.load_regional_table` |
| Table 5, Table S4 | `Results/Table_C1_vs_C2_gap.csv` |
| Tables S5, S6 | `Results/{FA,MD}_B/{F10,M9}_gcomputation_by_roi_summary.csv` |
| Table S7 | `Results/Sensitivity_B/tier2_looic.csv` |
| Table S8 | `Results/Sensitivity_B/item12_proxy_validity_summary.csv` |
| Table S9 | `Results/Sensitivity_B/evalues_summary.csv` |
| Table S10 | `Results/Sensitivity_B/spatial_correlation_morans_i.csv` |
| Tables S11, S13 | `Results/Sensitivity_B/vif_summary.csv` |
| Table S12 | `Results/{FA,MD}_B/posterior_corr_myelin_cell_{FA,MD}.csv` |
| Table S14 | `Results/Sensitivity_B/within_roi_correlations.csv` |
| Table S15 | `Results/Sensitivity_B/{F3,M2}_S6_blockFE_summary.csv` |
| Table SM4 | `Results/{FA,MD}_B/*_diagnostics.txt`, `*_summary.csv` |
