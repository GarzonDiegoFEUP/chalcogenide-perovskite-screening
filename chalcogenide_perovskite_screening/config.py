from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
BANDGAP_DATA_DIR = DATA_DIR / "bandgap_semicon"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
CRYSTALLM_DATA_DIR = DATA_DIR / "crystaLLM"
SYNTHESIS_DATA_DIR = DATA_DIR / "synthesis_planning_data"
SUSTAINABILITY_DATA_DIR = DATA_DIR / "sustainability_data"

MODELS_DIR = PROJ_ROOT / "models"
TREES_DIR = MODELS_DIR / "trees"
RESULTS_DIR = MODELS_DIR / "results"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SYNTHESIS_FIG_DIR = FIGURES_DIR / "synthesis_planning"

NOTEBOOKS_DIR = PROJ_ROOT / "notebooks"
FIGURES_NOTEBOOK_DIR = NOTEBOOKS_DIR / "figures/lc_data"
MODEL_CRABNET_DIR = NOTEBOOKS_DIR / "models"
TRAINED_MODELS = MODEL_CRABNET_DIR / "trained_models"

# Random seed
RANDOM_SEED = 187636123

# Chalcogenide/halide ionic radii mappings (rX values in picometers)
# Symbol to radii mapping
ANION_RADII = {
    'F': 133.0,
    'Cl': 181.0,
    'Se': 198.0,
    'Br': 196.0,
    'S': 184.0,
    'I': 220.00000000000003
}

# Radii to symbol mapping (inverse)
RADII_TO_ANION = {v: k for k, v in ANION_RADII.items()}

# Primary features to use

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
