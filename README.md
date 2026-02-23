# Chalcogenide-perovskite-screening

**ML-guided screening of chalcogenide perovskites as solar energy materials**

[![Python 3.8.2](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](CITATION.cff)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18701742.svg)](https://doi.org/10.5281/zenodo.18701742)

This repository contains the datasets, analysis notebooks, and source code for the paper:

> D. A. Garzón, L. Himanen, L. Andrade, S. Sadewasser, J. A. Márquez,
> *"ML-guided screening of chalcogenide perovskites as solar energy materials"* (2026).
>
> **Preprint coming soon.**

## Overview

Chalcogenide perovskites (ABX₃, X = S²⁻, Se²⁻) are promising absorber materials for
next-generation photovoltaic devices. This work presents a fully data-driven screening
pipeline that integrates:

1. **SISSO-derived tolerance factor (τ\*)** — an interpretable analytical descriptor
   ([Ouyang et al., 2018](https://doi.org/10.1103/PhysRevMaterials.2.083802)) for
   perovskite structural stability, outperforming the Goldschmidt tolerance factor
   ([Goldschmidt, 1926](https://doi.org/10.1007/BF01507527);
   [Bartel et al., 2019](https://doi.org/10.1126/sciadv.aav0693)) on experimental data.
2. **CrystaLLM crystal structure generation** — generative prediction of crystal structures
   ([Antunes et al., 2024](https://doi.org/10.1038/s41467-024-54639-7)) to validate
   corner-sharing perovskite-type topology.
3. **CrabNet bandgap estimation** — composition-based prediction of bandgaps
   ([Wang et al., 2021](https://doi.org/10.1038/s41524-021-00545-1)) trained on
   experimental halide perovskite and chalcogenide semiconductor data.
4. **Sustainability analysis** — multi-objective ranking using the Herfindahl–Hirschman
   Index (HHI) ([USGS, 2025](https://doi.org/10.5066/P13XCP3R)), ESG scores
   ([World Bank, 2023](https://datacatalog.worldbank.org/search/dataset/0037651)),
   and supply risk metrics.
5. **Experimental plausibility assessment** — crystal-likeness scoring via a pre-trained GCNN model
   ([Gu et al., 2022](https://doi.org/10.1038/s41524-022-00757-z);
   [Jang et al., 2020](https://doi.org/10.1021/jacs.0c07384)) for synthesizability
   likelihood.

## Installation

**Requirements:** Python <= 3.8.20.

### Using uv (recommended)

[uv](https://docs.astral.sh/uv/) manages the virtual environment, lockfile, and
dependencies in a single command:

```bash
# Install uv (if not already available)
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# or: brew install uv

# Clone and install
git clone https://github.com/GarzonDiegoFEUP/chalcogenide-perovskite-screening.git
cd chalcogenide-perovskite-screening
uv sync --extra dev --extra notebooks   # creates .venv, resolves & installs everything
```

Or equivalently via Make:

```bash
make install
```

### Using pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,notebooks]"
```

### Frozen lockfile (exact reproduction)

```bash
pip install -r requirements.txt
```

### SISSO feature derivation (optional)

The SISSO features are cached in `data/interim/features_sisso.csv`, so most users
**do not** need `sissopp`. If you want to re-derive features from scratch, install
[sissopp](https://github.com/rouyang2017/SISSO) manually **after** setting up the
environment:

### Environment variables

Some notebooks query the [Materials Project API](https://materialsproject.org/api).
Create a `.env` file in the project root:

```
MP_API_KEY=your_materials_project_api_key
```

### Running the notebooks

After installation, select the `.venv` kernel in VS Code (or register it manually):

```bash
uv run python -m ipykernel install --user --name chalc-pvk-sc --display-name "Python (chalc-pvk-sc)"
```

## Pipeline Notebooks

The analysis is organized as a sequential pipeline. Run notebooks in order:

| # | Notebook | Description |
|---|----------|-------------|
| 0 | `0_figures_paper.ipynb` | Generate all publication figures |
| 1 | `1_get_SISSO_features.ipynb` | Dataset creation, SISSO feature generation, tolerance factor training and evaluation, Platt scaling, compositional screening |
| 2 | `2_CrystaLLM_analysis.ipynb` | Parse CrystaLLM-generated CIF files, crystal-likeness scoring, corner-sharing vs edge-sharing classification |
| 3 | `3_Experimental_likelihood.ipynb` | GCNN-based synthesizability assessment and experimental plausibility scoring |
| 4 | `4_bandgap_prediction.ipynb` | CrabNet bandgap model: training, evaluation, predictions for all candidates |
| 4.1 | `4_1_encoder_comparison.ipynb` | Compare elemental encoding strategies for CrabNet (Pettifor vs default) |
| 4.2 | `4_2_data_size.ipynb` | Training data size ablation: effect on CrabNet bandgap prediction accuracy |
| 5 | `5_HHI_calculation.ipynb` | Sustainability analysis: HHI, supply risk, ESG scoring |


## Project Organization

```
├── LICENSE
├── Makefile                 <- Convenience commands (make data, make lint, make format)
├── README.md
├── pyproject.toml           <- Package metadata and abstract dependencies
├── requirements.txt         <- Frozen dependency lockfile for exact reproduction
│
├── data/
│   ├── raw/                 <- Original immutable data (radii, atomic features, datasets)
│   ├── interim/             <- Intermediate transformed data (pickled models, SISSO features)
│   ├── processed/           <- Final canonical datasets (train/test splits, results)
│   ├── crystaLLM/           <- CrystaLLM-generated CIF files and analysis results
│   ├── sustainability_data/ <- ESG, MCS, HHI, and earth abundance data
│   └── synthesis_planning_data/ <- Materials Project entries and reaction results
│
├── models/
│   ├── trees/               <- Decision tree visualizations
│   └── results/             <- Processed datasets and accuracy comparisons
│
├── notebooks/               <- Jupyter notebooks (numbered for ordering)
│   └── models/              <- Trained CrabNet model checkpoints
│
├── reports/
│   └── figures/             <- Generated publication figures
│
├── references/              <- Data dictionaries, manuals, explanatory materials
│
└── chalcogenide_perovskite_screening/  <- Source code (Python package)
    ├── __init__.py
    ├── config.py            <- Path configuration and constants
    ├── dataset.py           <- Data loading, cleaning, composition generation
    ├── features.py          <- SISSO feature engineering and PCA
    ├── plots.py             <- Visualization functions
    ├── modeling/            <- Machine learning models and training
    │   ├── __init__.py
    │   ├── train.py         <- Tolerance factor training and evaluation
    │   ├── predict.py       <- Model inference and predictions
    │   ├── GCCN_Predict.py  <- GCNN-based synthesizability prediction
    │   ├── CrabNet/         <- CrabNet bandgap model integration
    │   └── gcnn/            <- GCNN model implementation
    └── synthesis_planning/  <- Synthesis pathway optimization (adapted from Chen et al.)
        ├── __init__.py
        ├── README.md
        ├── LICENSE
        ├── synthesis_pathways.py
        ├── reactions.py
        ├── materials_entries.py
        ├── settings.py
        └── interfacial_pdplotter.py
```

## References

Key methods and data sources used in this pipeline:

| Step | Reference | DOI |
|------|-----------|-----|
| SISSO | Ouyang, R. et al. *Phys. Rev. Materials* **2**, 083802 (2018) | [10.1103/PhysRevMaterials.2.083802](https://doi.org/10.1103/PhysRevMaterials.2.083802) |
| Tolerance factor (τ) | Bartel, C. J. et al. *Science Advances* **5**, eaav0693 (2019) | [10.1126/sciadv.aav0693](https://doi.org/10.1126/sciadv.aav0693) |
| Goldschmidt factor | Goldschmidt, V. M. *Naturwissenschaften* **14**, 477–485 (1926) | [10.1007/BF01507527](https://doi.org/10.1007/BF01507527) |
| CrystaLLM | Antunes, L. M. et al. *Nature Communications* **15**, 10570 (2024) | [10.1038/s41467-024-54639-7](https://doi.org/10.1038/s41467-024-54639-7) |
| CrabNet | Wang, A. Y.-T. et al. *npj Computational Materials* **7**, 77 (2021) | [10.1038/s41524-021-00545-1](https://doi.org/10.1038/s41524-021-00545-1) |
| HHI / mineral data | U.S. Geological Survey. *Mineral Commodity Summaries 2025* | [10.5066/P13XCP3R](https://doi.org/10.5066/P13XCP3R) |
| ESG data | World Bank. *Environment, Social and Governance Data* (2023) | [Data Catalog](https://datacatalog.worldbank.org/search/dataset/0037651) |
| Synthesizability (GCNN) | Gu, G. H. et al. *npj Computational Materials* **8**, 71 (2022) | [10.1038/s41524-022-00757-z](https://doi.org/10.1038/s41524-022-00757-z) |
| Synthesizability | Jang, J. et al. *J. Am. Chem. Soc.* **142**, 18836–18843 (2020) | [10.1021/jacs.0c07384](https://doi.org/10.1021/jacs.0c07384) |

## Citation

If you use this software or data, please cite it using the metadata in [`CITATION.cff`](CITATION.cff):

```bibtex
@software{garzon2026chalc_screening,
  author  = {Garz{\'o}n, Diego A. and Himanen, Lauri and Andrade, Luisa
             and Sadewasser, Sascha and M{\'a}rquez, Jos{\'e} A.},
  title   = {{chalcogenide-perovskite-screening}},
  version = {1.0.1},
  year    = {2026},
  doi     = {10.5281/zenodo.18743650},
  url     = {https://doi.org/10.5281/zenodo.18743650},
  license = {MIT},
}
```

<!-- TODO: add paper BibTeX once published -->

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgements

DAG acknowledges the support by FCT — Fundação para a Ciência e Tecnologia, I.P.
(project ref. 2023.00258.BD). Authors acknowledge the COST Action "Emerging Inorganic
Chalcogenides for Photovoltaics (RENEW-PV)", CA21148.

