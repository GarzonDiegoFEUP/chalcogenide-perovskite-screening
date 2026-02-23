"""Generate interactive Plotly HTML figures for the TF-ChPVK-PV documentation.

Run from the repository root (or any directory) with:

    python docs/generate_interactive_figures.py

All HTML files are written to docs/assets/figures/ and embedded in
docs/interactive-figures.md via <iframe> tags.

Each file uses ``include_plotlyjs='cdn'`` so files stay small (~KB); an
internet connection is required to view them (or swap to 'inline' for
fully self-contained files, ~3 MB each).
"""

from pathlib import Path
import sys

# Ensure the package is importable when running the script directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from tf_chpvk_pv.config import PROCESSED_DATA_DIR, RESULTS_DIR
from tf_chpvk_pv.plots import (
    colormap_radii_interactive,
    plot_matrix_interactive,
    pareto_front_interactive,
    corr_matrix_interactive,
    normalize_abx3,
    plot_tau_star_histogram_interactive,
    plot_t_star_histogram_interactive,
    plot_t_star_vs_p_t_sisso_interactive,
)

OUTPUT_DIR = (Path(__file__).parent / "assets" / "figures").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOTLYJS = "cdn"  # swap to 'inline' for fully self-contained HTML files


def _save(fig, name: str) -> Path:
    out = OUTPUT_DIR / name
    fig.write_html(str(out), include_plotlyjs=PLOTLYJS)
    print(f"  Saved: {out.relative_to(ROOT)}")
    return out


# ---------------------------------------------------------------------------
# 1. Load shared data (mirrors the notebook cells in 0_figures_paper.ipynb)
# ---------------------------------------------------------------------------
print("Loading processed datasets …")

df_exp = pd.read_csv(PROCESSED_DATA_DIR / "chpvk_dataset.csv")
df_exp.set_index("material", inplace=True)

df_valid = pd.read_csv(PROCESSED_DATA_DIR / "valid_new_compositions.csv")
df_valid.rename(columns={"Unnamed: 0": "formula"}, inplace=True)

df_sisso = pd.read_csv(PROCESSED_DATA_DIR / "results_SISSO_with_bandgap.csv")
df_crystal = pd.read_csv(PROCESSED_DATA_DIR / "results_CrystaLLM_with_bandgap.csv")
df_hhi = pd.read_csv(PROCESSED_DATA_DIR / "results_SISSO_with_HHI.csv")
df_CLscore = pd.read_csv(PROCESSED_DATA_DIR / "final_CL_scores.csv")

# Normalise CL score columns
df_CLscore.rename(columns={"Composition": "formula"}, inplace=True)
df_CLscore["formula"] = (
    df_CLscore["formula"].str.replace(" ", "").str.replace("1", "")
)
df_hhi.rename(columns={"material": "formula"}, inplace=True)

# Build the merged candidate dataframe
df_sisso["norm_formula"] = df_sisso["formula"].apply(normalize_abx3)
df_crystal["norm_formula"] = df_crystal["formula"].apply(normalize_abx3)

df_crystal_sisso = df_sisso[df_sisso["norm_formula"].isin(df_crystal["norm_formula"])]

df_crystal_sisso_hhi = df_crystal_sisso[
    ["formula", "A", "B", "X", "bandgap", "bandgap_sigma", "t_sisso"]
].copy()
df_crystal_sisso_hhi = df_crystal_sisso_hhi.merge(
    df_hhi[["formula", "HHI", "SR"]], on="formula", how="left"
)
df_crystal_sisso_hhi_cl = df_crystal_sisso_hhi.merge(
    df_CLscore[["formula", "CL score", "CL score std"]], on="formula", how="left"
)

#Correct formula column (reconstruct from A, B, X) for all three dataframes to ensure consistency
for df in [df_crystal_sisso, df_crystal_sisso_hhi, df_crystal_sisso_hhi_cl]:
    df['formula'] = df['A'] + df['B'] + df['X'] + "3"
    df.loc[df['formula'].str.contains("Cu"), 'formula'] = 'Cu' + df.loc[df['formula'].str.contains("Cu"), 'formula'].str.replace("Cu", "")

# ---------------------------------------------------------------------------
# 2. Figure 2 — data distribution (tau*/t* histograms and t* vs P(tau*))
# ---------------------------------------------------------------------------
print("\n[Fig 2] Loading concat dataset for distribution plots …")
df_fig2 = pd.read_csv(RESULTS_DIR / "processed_chpvk_concat_dataset.csv")
df_fig2.rename(columns={"t_jess": "t*", "t_sisso": "tau*", "p_t_sisso": "p_tau*"}, inplace=True)

print("[Fig 2a] tau* histogram …")
fig = plot_tau_star_histogram_interactive(threshold=0.846, df=df_fig2)
_save(fig, "tau_star_histogram.html")

print("[Fig 2b] t* (Jess et al.) histogram …")
fig = plot_t_star_histogram_interactive(thresholds=[0.84, 1.02], df=df_fig2)
_save(fig, "t_star_histogram.html")

print("[Fig 2c] t* vs P(tau*) scatter …")
fig = plot_t_star_vs_p_t_sisso_interactive(df=df_fig2, thresholds=[0.84, 1.02])
_save(fig, "t_star_vs_p_tau_star.html")

# ---------------------------------------------------------------------------
# 3. Figure 3 — colormap_radii (stability heatmap over rA/rB space)
# ---------------------------------------------------------------------------
print("\n[Fig 3] Colormap radii (P(τ*)) …")
for anion in ["S", "Se"]:
    fig = colormap_radii_interactive(df=df_valid, exp_df=df_exp, t_sisso=False, anion=anion)
    _save(fig, f"colormap_radii_prob_{anion}.html")

print("[Fig 3alt] Colormap radii (τ* raw) …")
for anion in ["S", "Se"]:
    fig = colormap_radii_interactive(df=df_valid, exp_df=df_exp, t_sisso=True, anion=anion)
    _save(fig, f"colormap_radii_tsisso_{anion}.html")

# ---------------------------------------------------------------------------
# 3. Figure 5 — plot_matrix (element×element bandgap matrix)
# ---------------------------------------------------------------------------
df_sisso_bg = pd.read_csv(PROCESSED_DATA_DIR / "results_SISSO_with_bandgap.csv")
df_crystal_bg = pd.read_csv(PROCESSED_DATA_DIR / "results_CrystaLLM_with_bandgap.csv")
df_sisso_bg.rename(columns={"bandgap": "Eg"}, inplace=True)
df_crystal_bg.rename(columns={"bandgap": "Eg"}, inplace=True)

for anion in ["S", "Se"]:
    print(f"\n[Fig 5] plot_matrix — Eg, anion={anion} …")
    fig = plot_matrix_interactive(df_sisso_bg, df_crystal_bg, anion=anion, parameter="Eg")
    _save(fig, f"plot_matrix_Eg_{anion}.html")

# Figure S2 — element×element P(τ*) matrix
df_possible = pd.read_csv(PROCESSED_DATA_DIR / "valid_new_compositions.csv")
df_possible.rename(columns={"Unnamed: 0": "formula"}, inplace=True)

for anion in ["S", "Se"]:
    print(f"\n[Fig S2] plot_matrix — p_t_sisso, anion={anion} …")
    fig = plot_matrix_interactive(df_possible, df_crystal_bg, anion=anion, parameter="p_t_sisso")
    _save(fig, f"plot_matrix_prob_{anion}.html")

# ---------------------------------------------------------------------------
# 4. Figure 6 — pareto_front_plot (SR vs bandgap deviation)
# ---------------------------------------------------------------------------
print("\n[Fig 6a] Pareto front — SR, single junction (1.34 eV) …")
fig = pareto_front_interactive(df_crystal_sisso_hhi, variable="SR", Eg_ref=1.34)
_save(fig, "pareto_SR_sj.html")

print("[Fig 6b] Pareto front — SR, tandem (1.71 eV) …")
fig = pareto_front_interactive(df_crystal_sisso_hhi, variable="SR", Eg_ref=1.71)
_save(fig, "pareto_SR_tandem.html")

# 3-objective: CLS vs SR vs Eg
print("[Fig 7] Pareto 3‑front — CLS vs SR, coloured by Eg …")
df_crystal_sisso_hhi_cl["1-CL score"] = 1 - df_crystal_sisso_hhi_cl["CL score"]
fig_sj = pareto_front_interactive(
    df_crystal_sisso_hhi_cl, variable="1-CL score", Eg_ref=1.34
)
_save(fig_sj, "pareto_CLS_sj.html")

fig_t = pareto_front_interactive(
    df_crystal_sisso_hhi_cl, variable="1-CL score", Eg_ref=1.71
)
_save(fig_t, "pareto_CLS_tandem.html")

# ---------------------------------------------------------------------------
# 5. Figure S4 — correlation matrix
# ---------------------------------------------------------------------------
print("\n[Fig S4] Spearman correlation matrix …")
metrics = ["bandgap", "t_sisso", "SR", "CL score"]
dict_labels = {
    "bandgap": "Eₘ (eV)",
    "t_sisso": "τ*",
    "SR": "SR",
    "CL score": "CLS",
}
fig = corr_matrix_interactive(df_crystal_sisso_hhi_cl, metrics, dict_labels)
_save(fig, "corr_matrix.html")

print(f"\nDone. All figures saved to {OUTPUT_DIR.relative_to(ROOT)}")
