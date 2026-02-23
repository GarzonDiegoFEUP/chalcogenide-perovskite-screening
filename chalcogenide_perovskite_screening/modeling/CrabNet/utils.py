from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch
from loguru import logger
import pandas as pd
import numpy as np

from chalcogenide_perovskite_screening.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, TRAINED_MODELS, CRYSTALLM_DATA_DIR

# ---------------------------------------------------------------------------
# Device auto-detection: cuda > mps > cpu
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
logger.info(f"CrabNet device: {DEVICE}")


def get_raw_data(input_path_pvk: Path = RAW_DATA_DIR / "perovskite_bandgap_devices.csv",
             input_path_chalcogenides: Path = RAW_DATA_DIR / "chalcogenides_bandgap_devices.csv",
             input_path_chalc_semicon: Path = RAW_DATA_DIR / "chalcogen_semicon_bandgap.csv",
             output_path: Path = PROCESSED_DATA_DIR / "chpvk_dataset.csv",
             new_radii_path: Path = RAW_DATA_DIR / "Expanded_Shannon_Effective_Ionic_Radii.csv",
             turnley_radii_path: Path = RAW_DATA_DIR / "Turnley_Ionic_Radii.xlsx",) -> pd.DataFrame:
    """Combine and process bandgap data from multiple sources for CrabNet training.

    Merges perovskite device data, chalcogenide perovskite data, and general
    chalcogenide semiconductor bandgap data. Filters for valid bandgaps > 1 eV
    and removes outliers using median-based filtering.

    Args:
        input_path_pvk: Path to halide perovskite bandgap data.
        input_path_chalcogenides: Path to chalcogenide perovskite bandgap data.
        input_path_chalc_semicon: Path to chalcogenide semiconductor bandgap data.
        output_path: Path for processed output (not used in current implementation).
        new_radii_path: Path to expanded Shannon ionic radii data.
        turnley_radii_path: Path to Turnley ionic radii Excel file.

    Returns:
        pd.DataFrame: Combined and filtered bandgap dataset with source labels.
    """

    
    df_pvk = pd.read_csv(input_path_pvk)
    df_chalcogenides = pd.read_csv(input_path_chalcogenides)
    df_chalc_semicon = pd.read_csv(input_path_chalc_semicon)

    for df_, sc in zip([df_pvk, df_chalcogenides, df_chalc_semicon], ['pvk', 'chalcogenides', 'chalc_semicon']):
        df_['source'] = sc
    df = pd.concat([df_pvk, df_chalcogenides, df_chalc_semicon]).reset_index(drop=True)
    sources = (x*100/df.shape[0] for x in [df_pvk.shape[0], df_chalcogenides.shape[0], df_chalc_semicon.shape[0]])
    txt = 'The data comes from the following sources:\n{0:.2f} % from halide perovskites,\n{1:.4f} % from chalcogenides perovskites,\n{2:.2f} % from chalcogenide semiconductors'.format(*sources)
    print(txt)

    df = df[df['bandgap'].notna()]
    df = df[df['reduced_formulas'].notna()]
    df = df[df['bandgap'] > 1.0]

    #delete outliers with values below the median - 0.3
    from crabnet.utils.data import groupby_formula

    df_ = df.copy()

    # Rename the column 'bandgap' to 'target', and 'reduced_formula' to 'formula'
    df_.rename(columns={'bandgap': 'target'}, inplace=True)
    df_.rename(columns={'reduced_formulas': 'formula'}, inplace=True)

    # Group repeated formulas and take the median of the target
    df_grouped_formula = groupby_formula(df_, how='median')

    for idx in df_grouped_formula['index']:
      df.loc[idx, 'median_bandgap'] = df_grouped_formula.loc[df_grouped_formula['index'] == idx, 'target'].values[0]

    df['difference_from_median'] = abs(df['median_bandgap'] - df['bandgap'])
    df = df[df['difference_from_median'] <= 0.3]
    df.drop(columns=['median_bandgap', 'difference_from_median'], inplace=True)
    
    return df

def save_processed_data(df: pd.DataFrame,
                        output_path: Path = INTERIM_DATA_DIR / 'df_grouped_formula_complete_dataset.csv',):
    """Save processed DataFrame to CSV file with logging.

    Args:
        df: DataFrame to save.
        output_path: Path for the output CSV file.
    """
    df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")


def get_pettifor_features(df_grouped_formula,
                          input_pettifor_path: Path = RAW_DATA_DIR / 'pettifor_embedding.csv',
                          train=True,
                          original_df: pd.DataFrame = None,):
  """Compute composition-weighted Pettifor fingerprints and optionally split."""
  from pymatgen.core import Composition
  from ase.atom import Atom
  from sklearn.model_selection import train_test_split

  pettifor = pd.read_csv(input_pettifor_path, index_col=0)

  def get_onehot_comp(composition, elemental_embeddings):
    if isinstance(composition, str):
      composition = Composition(composition)
    a = composition.fractional_composition.get_el_amt_dict()
    comp_finger =  np.array([a.get(Atom(i).symbol, 0) for i in range(1,99)])
    comp_finger = comp_finger @ elemental_embeddings.values
    return comp_finger

  df_grouped_formula['pettifor'] = df_grouped_formula.formula.apply(lambda x: get_onehot_comp(x, pettifor))

  df = df_grouped_formula.copy()
  size_pf = df.pettifor.iloc[0].shape[0]
  feature_names = ['pettifor_' + str(i) for i in range(0, size_pf)]
  new_df = pd.DataFrame(columns = feature_names, index=df.index)

  for idx, arr in enumerate(df.pettifor.values):
      new_df.iloc[idx] = arr

  df = pd.concat([df, new_df.astype('float64')], axis=1)
  df.drop(columns=['pettifor'], inplace=True)

  if train:

    #add_source to do the separation
    if 'source' not in df.columns and original_df is not None:
     for formula in df['formula']:
        df.loc[df['formula'] == formula, 'source'] = original_df.loc[original_df['formula'] == formula, 'source'].values[0]

    try:
      # First split: 80% train, 20% temp (val+test), stratified by source
      train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df['source'], random_state=42
      )
      # Second split: split temp into 50/50 -> 10% val, 10% test, stratified by source
      val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['source'], random_state=42
      )
    except ValueError:
      # Fallback to non-stratified shuffle split if stratification is not possible
      train_df, val_df, test_df = np.split(
        df.sample(frac=1, random_state=42),
        [
            int(0.8 * len(df)),
            int(0.9 * len(df))
        ]
      )
    
    if 'source' in train_df.columns:
      train_df.drop(columns=['source'], inplace=True)
      val_df.drop(columns=['source'], inplace=True)
      test_df.drop(columns=['source'], inplace=True)

    return train_df, val_df, test_df, feature_names
  else:
    return df
  

def load_model(model_path: Path = TRAINED_MODELS / 'perovskite_bg_prediction.pth'):
    """Load a pre-trained CrabNet model, mapping to the best available device."""
    from crabnet.crabnet_ import CrabNet  # type: ignore
    from crabnet.kingcrab import SubCrab  # type: ignore

    sub_crab_model = SubCrab()
    crabnet_model = CrabNet()
    crabnet_model.model = sub_crab_model
    crabnet_model.load_network(str(model_path))
    crabnet_model.to(DEVICE)
    return crabnet_model

def get_test_r2_score_by_source_data(df: pd.DataFrame, original_df: pd.DataFrame,
                                     feature_names: List[str],
                                     crabnet_bandgap: Optional[Any] = None,
                                     model_path: Path = TRAINED_MODELS / 'perovskite_bg_prediction.pth',) -> None:
    """Compute R^2 scores separately for each data source.

    Evaluates model performance on subsets of data grouped by source
    (halide perovskites, chalcogenide perovskites, chalcogenide semiconductors)
    to assess cross-domain generalization.

    Args:
        df: Test DataFrame with 'formula' column.
        original_df: Original DataFrame with 'source' column for grouping.
        feature_names: List of Pettifor feature column names.
        crabnet_bandgap: Pre-loaded CrabNet model; if None, loads from file.
        model_path: Path to CrabNet model checkpoint.
    """

    for formula in df['formula']:
        df.loc[df['formula'] == formula, 'source'] = original_df.loc[original_df['formula'] == formula, 'source'].values[0]

    sources = original_df['source'].unique().tolist()
    for source in sources:
        df_source = df[df['source'] == source]
        print(f'\nResults for source: {source} with data size {df_source.shape[0]}')
        if df_source.shape[0] > 0:
            test_r2_score(df_source,
                          feature_names,
                          crabnet_bandgap=crabnet_bandgap,
                          model_path=model_path)
        else:
            print('No data available for this source.')

def test_r2_score(df,
                  feature_names=None,
                  crabnet_bandgap=None,
                  model_path: Path = TRAINED_MODELS / 'perovskite_bg_prediction.pth',
                  plot: bool = True):
  """Evaluate a CrabNet model on *df* and return a metrics dict.

  Returns
  -------
  dict with keys: r2, mse, mae, actual (array), predicted (array), sigma (array)
  """
  from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

  if not crabnet_bandgap:
    crabnet_bandgap = load_model(model_path=model_path)

  df_zeros = pd.DataFrame({"formula": df['formula'], "target": [0.0]*len(df['formula'])})
  if feature_names:
     df_zeros = pd.concat([df_zeros, df[feature_names]], axis=1)

  df_predicted, df_predicted_sigma = crabnet_bandgap.predict(df_zeros, return_uncertainty=True)

  actual = np.asarray(df['target'])
  predicted = np.asarray(df_predicted)
  sigma = np.asarray(df_predicted_sigma)

  # Filter out NaN predictions (e.g. from unsupported elements in smaller encoders)
  mask = np.isfinite(predicted)
  if not mask.all():
    n_nan = (~mask).sum()
    print(f'Warning: {n_nan}/{len(predicted)} predictions are NaN (unsupported elements), filtering them out.')
    actual = actual[mask]
    predicted = predicted[mask]
    sigma = sigma[mask]

  r2  = r2_score(actual, predicted)
  mse = mean_squared_error(actual, predicted)
  mae = mean_absolute_error(actual, predicted)

  if plot:
    from crabnet.utils.figures import act_pred  # type: ignore
    act_pred(actual, predicted)

  print(f'R2 score: {r2:.4f}')
  print(f'MSE: {mse:.4f}')
  print(f'MAE: {mae:.4f} eV')

  return {'r2': r2, 'mse': mse, 'mae': mae,
          'actual': actual, 'predicted': predicted, 'sigma': sigma}

def predict_bandgap(formula, 
                     crabnet_model = None,
                     model_path: Path = TRAINED_MODELS / 'perovskite_bg_prediction.pth',):
    
  if not crabnet_model:
    crabnet_model = load_model(model_path=model_path)

  input_df = pd.DataFrame({"formula": [formula], "target": [0.0]})
  input_df = get_pettifor_features(input_df, train=False)
  prediction, prediction_sigma = crabnet_model.predict(input_df, return_uncertainty=True)
  return prediction, prediction_sigma


def get_CrystaLLM_predictions(crabnet_model: Optional[Any] = None,
                               input_data_CrystaLLM: Path = CRYSTALLM_DATA_DIR / 'results_crystallm.csv',
                               output_data_CrystaLLM: Path = PROCESSED_DATA_DIR / 'results_CrystaLLM_with_bandgap.csv') -> pd.DataFrame:
    """Predict bandgaps for CrystaLLM-generated compositions.

    Args:
        crabnet_model: Pre-loaded CrabNet model; if None, loads from file.
        input_data_CrystaLLM: Path to CrystaLLM results CSV.
        output_data_CrystaLLM: Path to save predictions CSV.

    Returns:
        pd.DataFrame: CrystaLLM compositions with 'bandgap' and 'bandgap_sigma' columns.
    """

    if not crabnet_model:
        crabnet_model = load_model()

    df_compositions = pd.read_csv(input_data_CrystaLLM)
    df_compositions.rename(columns={'material': 'formula'}, inplace=True)
    df_compositions.set_index('formula', inplace=True)
    for formula in df_compositions.index:
        prediction, prediction_sigma = predict_bandgap(formula, crabnet_model)
        df_compositions.loc[formula, 'bandgap'] = prediction
        df_compositions.loc[formula, 'bandgap_sigma'] = prediction_sigma

    df_compositions.to_csv(output_data_CrystaLLM)

    return df_compositions

def get_SISSO_predictions(crabnet_model: Optional[Any] = None,
                          input_data_SISSO: Path = PROCESSED_DATA_DIR / 'stable_compositions.csv',
                          output_data_SISSO: Path = PROCESSED_DATA_DIR / 'results_SISSO_with_bandgap.csv') -> pd.DataFrame:
    """Predict bandgaps for SISSO-selected stable compositions.

    Adds bandgap predictions with uncertainties to compositions identified
    as stable by the t_sisso tolerance factor screening.

    Args:
        crabnet_model: Pre-loaded CrabNet model; if None, loads from file.
        input_data_SISSO: Path to SISSO stable compositions CSV.
        output_data_SISSO: Path to save predictions CSV.

    Returns:
        pd.DataFrame: Stable compositions with 'bandgap' and 'bandgap_sigma' columns.
    """

    if not crabnet_model:
        crabnet_model = load_model()

    df_compositions = pd.read_csv(input_data_SISSO)
    df_compositions.rename(columns={'Unnamed: 0': 'formula'}, inplace=True)
    df_compositions.set_index('formula', inplace=True)
    for formula in df_compositions.index:
        prediction, prediction_sigma = predict_bandgap(formula, crabnet_model)
        df_compositions.loc[formula, 'bandgap'] = prediction
        df_compositions.loc[formula, 'bandgap_sigma'] = prediction_sigma

    df_compositions.to_csv(output_data_SISSO)

    return df_compositions

def get_experimental_predictions(crabnet_model: Optional[Any] = None,
                                 input_data_experimental: Path = RAW_DATA_DIR / 'chalcogenides_bandgap_devices.csv',
                                 output_data_experimental: Path = PROCESSED_DATA_DIR / 'results_experimental_with_bandgap.csv',) -> pd.DataFrame:
    """Predict bandgaps for experimental chalcogenide formulas and compare.

    Evaluates CrabNet predictions against experimental bandgap measurements
    for known chalcogenide perovskite compounds, printing comparisons and
    saving results.

    Returns:
        pd.DataFrame: Experimental data with 'predicted_bandgap' and
            'predicted_bandgap_sigma' columns added.
    """

    if not crabnet_model:
        crabnet_model = load_model()

    df_chalcogenides = pd.read_csv(input_data_experimental)
    for formula in df_chalcogenides.descriptive_formulas.unique():
        prediction, prediction_sigma = predict_bandgap(formula)
        print(f'Experimental bandgap for {formula}:', str(df_chalcogenides.loc[df_chalcogenides['descriptive_formulas'] == formula, 'bandgap'].values[0]) + ' eV')
        print(f'Bandgap prediction for {formula}:', f"{round(prediction[0], 2)} ± {round(prediction_sigma[0], 2)}" + ' eV')

        df_chalcogenides.loc[df_chalcogenides['descriptive_formulas'] == formula, 'predicted_bandgap'] = prediction[0]
        df_chalcogenides.loc[df_chalcogenides['descriptive_formulas'] == formula, 'predicted_bandgap_sigma'] = prediction_sigma[0]

    df_chalcogenides.to_csv(output_data_experimental)

    return df_chalcogenides


# ---------------------------------------------------------------------------
# Register Pettifor embedding as a CrabNet elem_prop
# ---------------------------------------------------------------------------
def register_pettifor_elem_prop(
    pettifor_path: Path = RAW_DATA_DIR / "pettifor_embedding.csv",
    force: bool = False,
) -> str:
    """Copy the Pettifor matrix into CrabNet's element_properties dir.

    This allows ``CrabNet(elem_prop='pettifor')`` to work like any built-in
    encoder (mat2vec, oliynyk, …).

    Parameters
    ----------
    pettifor_path : Path
        Path to the Pettifor embedding CSV (99×98 including header/index; 98×98 data matrix).
    force : bool
        Re-write even if the file already exists.

    Returns
    -------
    str
        The ``elem_prop`` name to pass to CrabNet (``'pettifor'``).
    """
    from os.path import join, dirname
    import crabnet.kingcrab as _kc

    elem_dir = join(dirname(_kc.__file__), "data", "element_properties")
    dest = Path(elem_dir) / "pettifor.csv"

    if dest.exists() and not force:
        logger.info(f"Pettifor elem_prop already registered at {dest}")
        return "pettifor"

    pet = pd.read_csv(pettifor_path, index_col=0)
    # Rename columns to generic V0..V97 to avoid clashing with CrabNet's
    # internal element symbol parsing.
    pet.columns = [f"V{i}" for i in range(pet.shape[1])]
    pet.to_csv(dest)
    logger.info(f"Registered Pettifor elem_prop → {dest}")
    return "pettifor"


# ---------------------------------------------------------------------------
# Deprecated alias — keeps old notebooks working until they're updated
# ---------------------------------------------------------------------------
get_petiffor_features = get_pettifor_features


