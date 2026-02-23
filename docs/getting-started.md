# Getting Started

This guide covers environment setup and installation so you can run the full screening pipeline.

## Prerequisites

- **Python <= 3.8.20** (required by dependency constraints)
- Git
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

!!! warning "Python version constraint"
    This project requires **Python <= 3.8.20** due to specific dependency version locks. Using a newer Python version will cause resolution failures.

## Installation

### Option 1: Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) manages the virtual environment, lockfile, and dependencies in a single command:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/GarzonDiegoFEUP/chalcogenide-perovskite-screening.git
cd chalcogenide-perovskite-screening

# Install all dependencies
uv sync --extra dev --extra notebooks
```

Or equivalently via Make:

```bash
make install
```

### Option 2: Using pip and venv

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -e ".[dev,notebooks]"
```

### Option 3: Frozen lockfile (exact reproduction)

For reproducible results matching the exact development environment:

```bash
pip install -r requirements.txt
```

!!! tip "Exact reproduction"
    The frozen lockfile in `requirements.txt` pins every transitive dependency to the exact versions used during development. Use this if you need bit-for-bit reproducibility.

## Configuration

### Materials Project API Key

Some notebooks query the [Materials Project API](https://materialsproject.org/api) for structural data. To enable this:

1. Get your API key from [Materials Project](https://materialsproject.org/api)
2. Copy the example environment file and fill in your key:

```bash
cp .env.example .env
# Edit .env and set: MP_API_KEY=your_api_key_here
```

!!! note "Optional dependency"
    The Materials Project API key is only required for notebooks that retrieve crystal structure data. The core pipeline will run without it using cached data in `data/`.

### SISSO Features (Optional)

SISSO features are pre-cached in `data/interim/features_sisso.csv`. If you need to re-derive them from scratch, install [sissopp](https://github.com/rouyang2017/SISSO) manually after setting up the environment.

## Jupyter Notebook Setup

Register the virtual environment as a Jupyter kernel:

```bash
uv run python -m ipykernel install --user --name tf-chpvk --display-name "Python (TF-ChPVK)"
```

Then select the **Python (TF-ChPVK)** kernel when opening notebooks in VS Code or Jupyter Lab.

## Next Steps

Once installed, head to the [Pipeline](pipeline.md) page to run the analysis notebooks in order.

## Project Structure

```
chalcogenide-perovskite-screening/
├── data/                    # Data directory (raw, interim, processed)
│   ├── raw/                 # Original immutable datasets
│   ├── interim/             # Intermediate transformed data
│   ├── processed/           # Final canonical datasets
│   ├── crystaLLM/           # CrystaLLM-generated CIF files
│   └── sustainability_data/ # ESG, HHI, earth abundance data
├── notebooks/               # Jupyter analysis notebooks (numbered)
├── chalcogenide_perovskite_screening/  # Python package
│   ├── config.py            # Path configuration and constants
│   ├── dataset.py           # Data loading and processing
│   ├── features.py          # SISSO feature engineering
│   ├── plots.py             # Visualization utilities
│   ├── modeling/            # ML models (GCNN, CrabNet, tolerance factor)
│   └── synthesis_planning/  # Synthesis pathway optimization
├── models/                  # Trained model weights and results
├── docs/                    # Documentation (MkDocs)
├── pyproject.toml           # Package metadata and dependencies
└── requirements.txt         # Frozen dependency lockfile
```
