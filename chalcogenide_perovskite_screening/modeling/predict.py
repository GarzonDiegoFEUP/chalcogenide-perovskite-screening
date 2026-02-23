from pathlib import Path
from typing import List, Tuple

import typer
from loguru import logger
from tqdm import tqdm
import pandas as pd
import pickle

from chalcogenide_perovskite_screening.config import MODELS_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()


@app.command()
def main():
    """CLI entry point for predicting stable compositions.

    Filters generated compositions using the t_sisso tolerance factor
    threshold to identify candidates predicted to form stable perovskites.
    """
    check_stable_compositions('t_sisso')


def check_stable_compositions(t: str, 
                              valid_new_compositions_data_path: Path = PROCESSED_DATA_DIR / "valid_new_compositions.csv",
                              tolerance_factor_dict_path: Path = INTERIM_DATA_DIR / "tolerance_factors.pkl",
                              output_path: Path = PROCESSED_DATA_DIR / "stable_compositions.csv") -> Tuple[List[str], pd.DataFrame]:
    """Filter compositions by tolerance factor threshold to predict stable perovskites.

    Applies the learned tolerance factor threshold to screen generated
    compositions, identifying those predicted to form stable perovskite
    structures based on their ionic radii and oxidation states.

    Args:
        t: Name of tolerance factor to use for screening (e.g., 't_sisso').
        valid_new_compositions_data_path: Path to CSV with generated compositions.
        tolerance_factor_dict_path: Path to pickle file containing tolerance
            factor expressions and thresholds.
        output_path: Path to save CSV of predicted stable compositions.

    Returns:
        tuple: Contains:
            - stable_candidates_t_sisso (list): Formula strings of stable candidates.
            - df_out (pd.DataFrame): Filtered DataFrame of stable compositions.
    """
    
    df = pd.read_csv(valid_new_compositions_data_path)

    with open(tolerance_factor_dict_path, 'rb') as file:
        tolerance_factor_dict = pickle.load(file)

    stable_candidates_t_sisso=[]
    for i in list(range(len(df))):
        if df[t][i] < tolerance_factor_dict[t][1]:
            idx = df.index[i]
            name_real = df.A[idx] + df.B[idx] + df.X[idx] + '3'
            stable_candidates_t_sisso.append(name_real)

    print('According to {}, {} ({}%) compositions are predicted to be stable as perovskites:'.format(t, len(stable_candidates_t_sisso), len(stable_candidates_t_sisso)/df.shape[0] * 100))
    print(stable_candidates_t_sisso)

    df_out = df[df[t] < tolerance_factor_dict[t][1]]

    if t == 't_sisso':
        df_out.to_csv(output_path, index=False)

    return stable_candidates_t_sisso, df_out


if __name__ == "__main__":
    app()
