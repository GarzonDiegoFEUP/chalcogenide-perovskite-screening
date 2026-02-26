from pathlib import Path
from typing import Any

import typer
from loguru import logger
from tqdm import tqdm
import numpy as np
from sklearn import tree, metrics, set_config
from sklearn.model_selection import cross_validate
set_config(enable_metadata_routing=True)
from sklearn.calibration import CalibratedClassifierCV 
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from chalcogenide_perovskite_screening.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR, TREES_DIR, RESULTS_DIR, RADII_TO_ANION

app = typer.Typer()


@app.command()
def main():
    """CLI entry point for model training and tolerance factor evaluation.

    Orchestrates the complete training pipeline:
    1. Trains decision trees on SISSO features to identify best tolerance factor
    2. Evaluates t_sisso expression on train/test data
    3. Tests multiple tolerance factors (t_sisso, t, tau, t_jess) with thresholds
    4. Trains Platt scaling for probability calibration
    """
    t_sisso_expression = train_tree_sis_features()
    logger.success("Modeling training complete.")

    logger.info("Evaluating t_sisso...")
    train_df, test_df, tolerance_factor_dict = evaluate_t_sisso(t_sisso_expression)

    tfs = ['t_sisso', 't', 'tau', 't_jess']
    tf_tresh = [1, 2, 1, 2]

    df_acc =pd.DataFrame()
    clfs = {}

    for tf, tresh in zip(tfs, tf_tresh):
        df_acc, clf_t = test_tolerance_factor(tf, train_df, test_df, tolerance_factor_dict, df_acc, n_tresh=tresh)
        clfs[tf] = clf_t

    df_acc.to_csv(RESULTS_DIR / 'tolerance factors accuracy.csv')

    logger.success("Modeling evaluation complete.")

    train_platt_scaling(train_df, test_df, clfs['t_sisso'])


def train_platt_scaling(train_df: pd.DataFrame, test_df: pd.DataFrame, clf_t: Any, t: str = 't_sisso',
                        output_dir: Path = RESULTS_DIR) -> tuple[pd.DataFrame, pd.DataFrame, CalibratedClassifierCV]:
    """Train Platt scaling model for probability calibration.

    Applies isotonic regression (CalibratedClassifierCV) to convert raw
    tolerance factor values into calibrated probability estimates P(t_sisso),
    enabling probabilistic stability predictions.

    Args:
        train_df: Training DataFrame with tolerance factor column.
        test_df: Test DataFrame with tolerance factor column.
        clf_t: Pre-trained decision tree classifier for initial predictions.
        t: Name of tolerance factor column to calibrate.
        output_dir: Directory to save processed datasets with probabilities.

    Returns:
        tuple: Contains:
            - train_df (pd.DataFrame): Training data with added 'p_{t}' column.
            - test_df (pd.DataFrame): Test data with added 'p_{t}' column.
            - clf2_sisso (CalibratedClassifierCV): Fitted Platt scaling model.
    """

    logger.info("Training Platt scaling model...")

    x_train_t_sisso = train_df[t].to_numpy()
    x_test_t_sisso = test_df[t].to_numpy()

    labels_platt=clf_t.predict(x_train_t_sisso.reshape(-1,1))
    clf2_sisso = CalibratedClassifierCV(cv=3)
    clf2_sisso = clf2_sisso.fit(x_train_t_sisso.reshape(-1,1), labels_platt)
    p_t_sisso_train=clf2_sisso.predict_proba(x_train_t_sisso.reshape(-1,1))[:,1]
    p_t_sisso_test=clf2_sisso.predict_proba(x_test_t_sisso.reshape(-1,1))[:,1]
    train_df['p_' + t] = p_t_sisso_train            # add p_t_sisso to the train and test data frame
    test_df['p_' + t] = p_t_sisso_test

    train_df.to_csv(output_dir / 'processed_chpvk_train_dataset.csv')
    test_df.to_csv(output_dir / 'processed_chpvk_test_dataset.csv')

    logger.success("Platt scaling model training complete.")

    return train_df, test_df, clf2_sisso


def train_tree_sis_features(
    features_path: Path = INTERIM_DATA_DIR / "features_sisso.csv",
    train_data_path: Path = PROCESSED_DATA_DIR / "chpvk_train_dataset.csv",
    test_data_path: Path = PROCESSED_DATA_DIR / "chpvk_test_dataset.csv",
) -> str:
    """Train decision tree classifiers on SISSO features to find best tolerance factor.

    Ranks all SISSO-generated features by classification accuracy using
    decision trees with cross-validation. The top-ranked feature expression
    becomes the t_sisso tolerance factor formula.

    Args:
        features_path: Path to SISSO-generated features CSV.
        train_data_path: Path to training dataset with exp_label.
        test_data_path: Path to test dataset.

    Returns:
        str: Mathematical expression for t_sisso tolerance factor,
            converted to Python eval-compatible syntax.
    """
    logger.info("Training tree model with SISSO features...")

    #train classification trees for the selected descriptors
    train_df = pd.read_csv(train_data_path, index_col=0)
    test_df = pd.read_csv(test_data_path, index_col=0)
    feature_df = pd.read_csv(features_path, index_col=0)



    #depth of the classification tree - user has to choose
    def rank_tree(labels, feature_space, depth):                       #rank features according to the classification-tree accuracy                             
        score = []
        for i in list(range(0,feature_space.shape[1])):  
            #print(feature_space.columns[i])              # 'i' is a column and 'for' is from the first column to the last one
            x=np.array(feature_space)[:,i]
            if (x > 1e6).any():
                print('Feature %s has values greater than 1e6. Skipping.' % feature_space.columns.values[i])
                continue                           # take the first column values
            else:
                clf = tree.DecisionTreeClassifier(max_depth=depth, class_weight='balanced', criterion='entropy')
                
                clf_cv = cross_validate(clf, x.reshape(-1,1), labels, scoring='accuracy')
                clf_cv_score = np.mean(clf_cv['test_score'])
                
                #clf = clf.fit(x.reshape(-1,1), labels)                     # Build a decision-tree classifier from the training set (X, y). X is the values of features (for each for iteration on column) and Y is the target value, here exp_label
                score.append([feature_space.columns.values[i],clf_cv_score])      # make a list of the feature and the mean accuracy of the all values of that feaure (for different materials)
        score_sorted=sorted(score,reverse=True,key=lambda x: x[1])     # sort the features based on the accuracy
        return score_sorted

    labels_train=train_df["exp_label"].to_numpy()
    labels_test=test_df["exp_label"].to_numpy()
    rank_list=rank_tree(labels_train, feature_df, 1)
    tree_pd =pd.DataFrame(rank_list, columns=['feature','tree accuracy'])       # make a new data frame of the feature and the accuracies

    # the first ranked feature is the t_sisso
    t_sisso_expression=str(rank_list[0][0])   
    t_sisso_expression = t_sisso_expression.replace("ln","log")
    t_sisso_expression = t_sisso_expression.replace("^","**")
    print('Identified expression for t_sisso: %s' % t_sisso_expression)
    return t_sisso_expression

def train_tree_sis_features_Ch(
    features_path: Path = INTERIM_DATA_DIR / "features_sisso.csv",
    train_data_path: Path = PROCESSED_DATA_DIR / "chpvk_train_dataset.csv",
    test_data_path: Path = PROCESSED_DATA_DIR / "chpvk_test_dataset.csv",
) -> str:
    """Train decision tree classifiers with sample weights for chalcogenides.

    Similar to train_tree_sis_features but applies higher weights to
    chalcogenide (S, Se) compounds during training and uses F1 score
    instead of accuracy for feature ranking.

    Args:
        features_path: Path to SISSO-generated features CSV.
        train_data_path: Path to training dataset with exp_label.
        test_data_path: Path to test dataset.

    Returns:
        str: Mathematical expression for t_sisso tolerance factor,
            optimized for chalcogenide classification.
    """
    logger.info("Training tree model with SISSO features...")

    #train classification trees for the selected descriptors
    train_df = pd.read_csv(train_data_path, index_col=0)
    test_df = pd.read_csv(test_data_path, index_col=0)
    feature_df = pd.read_csv(features_path, index_col=0)
    
    sample_weights = train_df['rX'].to_numpy()
    sample_weights = np.where(sample_weights == 198, 2, sample_weights)
    sample_weights = np.where(sample_weights == 184, 2, sample_weights)
    sample_weights = np.where(sample_weights != 2, 1, sample_weights)

    #depth of the classification tree - user has to choose
    def rank_tree(labels, feature_space, depth, sample_weights):                       #rank features according to the classification-tree accuracy                             
        score = []
        for i in list(range(0,feature_space.shape[1])):  
            #print(feature_space.columns[i])              # 'i' is a column and 'for' is from the first column to the last one
            x=np.array(feature_space)[:,i]
            if (x > 1e6).any():
                print('Feature %s has values greater than 1e6. Skipping.' % feature_space.columns.values[i])
                continue                           # take the first column values
            else:
                clf = tree.DecisionTreeClassifier(max_depth=depth, class_weight='balanced', criterion='entropy')
                clf.set_fit_request(sample_weight=True)
                clf_cv = cross_validate(clf, x.reshape(-1,1), labels, scoring='f1', params={'sample_weight': sample_weights})
                clf_cv_score = np.mean(clf_cv['test_score'])
                
                #clf = clf.fit(x.reshape(-1,1), labels)                     # Build a decision-tree classifier from the training set (X, y). X is the values of features (for each for iteration on column) and Y is the target value, here exp_label
                score.append([feature_space.columns.values[i],clf_cv_score])      # make a list of the feature and the mean accuracy of the all values of that feaure (for different materials)
        score_sorted=sorted(score,reverse=True,key=lambda x: x[1])     # sort the features based on the accuracy
        return score_sorted

    labels_train=train_df["exp_label"].to_numpy()
    labels_test=test_df["exp_label"].to_numpy()
    rank_list=rank_tree(labels_train, feature_df, 1, sample_weights=sample_weights)
    tree_pd =pd.DataFrame(rank_list, columns=['feature','tree accuracy'])       # make a new data frame of the feature and the accuracies

    # the first ranked feature is the t_sisso
    t_sisso_expression=str(rank_list[0][0])   
    t_sisso_expression = t_sisso_expression.replace("ln","log")
    t_sisso_expression = t_sisso_expression.replace("^","**")
    print('Identified expression for t_sisso: %s' % t_sisso_expression)
    return t_sisso_expression
    
def evaluate_t_sisso(t_sisso_expression: str, idx: int = -1,
                     train_df_path: Path = PROCESSED_DATA_DIR / "chpvk_train_dataset.csv",
                     test_df_path: Path = PROCESSED_DATA_DIR / "chpvk_test_dataset.csv") -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate tolerance factor expressions on train and test datasets.

    Computes t_sisso and reference tolerance factors (t, tau, t_jess) for
    all compounds in the datasets by evaluating their mathematical expressions.

    Args:
        t_sisso_expression: Mathematical expression string for t_sisso.
        idx: If not -1, append index to column name (e.g., 't_sisso_0').
        train_df_path: Path to training dataset CSV.
        test_df_path: Path to test dataset CSV.

    Returns:
        tuple: Contains:
            - train_df (pd.DataFrame): Training data with computed tolerance factors.
            - test_df (pd.DataFrame): Test data with computed tolerance factors.
            - tolerance_factor_dict (dict): Dictionary mapping factor names to
              [expression, threshold] lists.
    """
    
    import re
    
    if t_sisso_expression == '':
        tolerance_factor_dict = {
    "t": ["(rA+rX)/(1.41421*(rB+rX))"],
    "tau": ["rX/rB-nA*(nA-rA_rB_ratio/log(rA_rB_ratio))"],
    "t_jess": ["chi_AX_ratio * (rA+rX)/(1.41421*chi_BX_ratio*(rB+rX))"],
    }
    else:
        pattern = r"\(\|(.+?)\|\)"
        replacement = r"abs(\1)"
        t_sisso_expression = re.sub(pattern, replacement, t_sisso_expression)
        while '|' in t_sisso_expression:
            t_sisso_expression = re.sub(pattern, replacement, t_sisso_expression)
        if idx != -1:
            tolerance_factor_dict = {
        't_sisso_' + str(idx): [t_sisso_expression],
        "t": ["(rA+rX)/(1.41421*(rB+rX))"],
        "tau": ["rX/rB-nA*(nA-rA_rB_ratio/log(rA_rB_ratio))"],
        "t_jess": ["chi_AX_ratio * (rA+rX)/(1.41421*chi_BX_ratio*(rB+rX))"],
        }
        else:
            tolerance_factor_dict = {
        "t_sisso": [t_sisso_expression],
        "t": ["(rA+rX)/(1.41421*(rB+rX))"],
        "tau": ["rX/rB-nA*(nA-rA_rB_ratio/log(rA_rB_ratio))"],
        "t_jess": ["chi_AX_ratio * (rA+rX)/(1.41421*chi_BX_ratio*(rB+rX))"],
        }

    train_df = pd.read_csv(train_df_path, index_col=0)
    test_df = pd.read_csv(test_df_path, index_col=0)

    if 'rA_rB_ratio' not in train_df.columns:
        train_df.eval('rA_rB_ratio = rA/rB', inplace=True)
        test_df.eval('rA_rB_ratio = rA/rB', inplace=True)

    
    #make a dictionary for t_sisso,t, tau
    #tolerance_factor_dict = {
    #"t_sisso": [t_sisso_expression],
    #"t": ["(rA+rX)/(1.41421*(rB+rX))"],
    #"tau": ["rX/rB-nA*(nA-rA_rB_ratio/log(rA_rB_ratio))"],
    #"t_jess": ["chi_AX_ratio * (rA+rX)/(1.41421*chi_BX_ratio*(rB+rX))"],
    #"t_old": ["sqrt(chi_AX_ratio) * 1/log(rA_rB_ratio) - (rB_rX_ratio * nB) + (rA_rB_ratio/chi_AX_ratio)"],
    #}


    #Add tau threshold
    tolerance_factor_dict["tau"].append(4.18)
    #tolerance_factor_dict["t_old"].append(2.75)

    if t_sisso_expression != '':
        if idx == -1:
            train_df.eval('t_sisso = ' + tolerance_factor_dict['t_sisso'][0], inplace=True)
            test_df.eval('t_sisso = ' + tolerance_factor_dict['t_sisso'][0], inplace=True)
        else:
            train_df.eval('t_sisso_' + str(idx) + '= ' + tolerance_factor_dict['t_sisso_' + str(idx)][0], inplace=True)
            test_df.eval('t_sisso_' + str(idx) + '= ' + tolerance_factor_dict['t_sisso_' + str(idx)][0], inplace=True)


    train_df.eval('t = '+ tolerance_factor_dict['t'][0],inplace=True)
    train_df.eval('tau = '+ tolerance_factor_dict['tau'][0], inplace=True) 
    train_df.eval('t_jess = '+ tolerance_factor_dict['t_jess'][0], inplace=True) 
    #train_df.eval('t_old = '+ tolerance_factor_dict['t_old'][0], inplace=True) 

    test_df.eval('t = '+ tolerance_factor_dict['t'][0], inplace=True)
    test_df.eval('tau = '+ tolerance_factor_dict['tau'][0], inplace=True)
    test_df.eval('t_jess = '+ tolerance_factor_dict['t_jess'][0], inplace=True)
    #test_df.eval('t_old = '+ tolerance_factor_dict['t_old'][0], inplace=True)


    return train_df, test_df, tolerance_factor_dict

def test_tolerance_factor(t: str, train_df: pd.DataFrame, test_df: pd.DataFrame, tolerance_factor_dict: dict[str, Any], df_acc: pd.DataFrame = pd.DataFrame(),
                          n_tresh: int = 1, cl_w: str = 'balanced', crit: str = 'entropy') -> tuple[pd.DataFrame, Any]:
    """Test a tolerance factor using decision tree classification.

    Trains a decision tree on the tolerance factor values, determines optimal
    threshold(s), and computes accuracy metrics for train, test, and per-anion
    subsets. Saves a visualization of the decision tree.

    Args:
        t: Name of tolerance factor column to test.
        train_df: Training DataFrame with tolerance factor and exp_label.
        test_df: Test DataFrame with tolerance factor and exp_label.
        tolerance_factor_dict: Dictionary to store threshold values.
        df_acc: DataFrame to accumulate accuracy results across factors.
        n_tresh: Decision tree depth (1 for single threshold, 2 for range).
        cl_w: Class weight strategy for decision tree ('balanced' recommended).
        crit: Split criterion ('entropy' or 'gini').

    Returns:
        tuple: Contains:
            - df_acc (pd.DataFrame): Updated accuracy results DataFrame.
            - clf1_model: Fitted decision tree classifier.
    """
    
    labels_train=train_df["exp_label"].to_numpy()
    labels_test=test_df["exp_label"].to_numpy()
    
    if len(df_acc.columns) == 0:
        df_acc = pd.DataFrame(columns=['t_sisso', 't', 'tau', 't_jess'], index=['train_data', 'test_data', 'all_data'] )
        
    x_train_=train_df[t].to_numpy()
    x_test_=test_df[t].to_numpy()
    
    
    clf1_tree = tree.DecisionTreeClassifier(max_depth=n_tresh, class_weight=cl_w, criterion=crit)
    
    clf1_cv = cross_validate(clf1_tree, x_train_.reshape(-1,1), labels_train, scoring='accuracy')
    
    clf1_cv_score = np.mean(clf1_cv['test_score'])
    
    clf1_model = clf1_tree.fit(x_train_.reshape(-1,1),labels_train)
    labels_pred_=clf1_model.predict(x_test_.reshape(-1,1))
    tree.plot_tree(clf1_model)

    name_figure = 'tree_' + t + '.png'
    
    plt.savefig(TREES_DIR / name_figure)
    
    acc_train = clf1_model.score(x_train_.reshape(-1,1),labels_train)
    acc_test = metrics.accuracy_score(labels_test, labels_pred_)
    
    #General accuracy
    all_data = np.append(x_train_, x_test_)
    all_labels = np.append(labels_train, labels_test)
    labels_all_data = clf1_model.predict(all_data.reshape(-1,1))
    acc_all = metrics.accuracy_score(all_labels, labels_all_data)
    
    print('Classification tree accuracy (for ' + t + ') on the train set: %f.' % acc_train)
    print('Classification tree accuracy (for ' + t + ') on the train set (5 fold CV): %f.' % clf1_cv_score)
    print('Classification tree accuracy (for ' + t + ') on the test set: %f.' % acc_test)

    if n_tresh == 2:
        threshold_=[clf1_model.tree_.threshold[0],clf1_model.tree_.threshold[4]]

        # add t threshold to dictionary
        tolerance_factor_dict[t].append(threshold_)

        print('%f < ' % tolerance_factor_dict[t][1][0] +  t +' < %f indicates stable perovskites.' % tolerance_factor_dict[t][1][1])
    elif n_tresh == 1:
        threshold_= clf1_model.tree_.threshold[0]

        #Add threshold to the dictionary
        tolerance_factor_dict[t].append(threshold_)

        print(t + ' < %f indicates stable perovskites.' % tolerance_factor_dict[t][1])
              
    df_acc.loc['train_data', t] = acc_train
    df_acc.loc['test_data', t] = acc_test
    df_acc.loc['all_data', t] = acc_all
    
    #get accuracy per X anion with the rX
    dict_ch = RADII_TO_ANION
  
        
    for rx in train_df.rX.unique():
        x_test_ch = test_df.loc[test_df.rX == rx, t].to_numpy()
        x_train_ch = train_df.loc[train_df.rX == rx, t].to_numpy()
        
        labels_test_ch = labels_test[test_df.rX == rx]
        labels_train_ch = labels_train[train_df.rX == rx]
        
        labels_pred_ch=clf1_model.predict(x_test_ch.reshape(-1,1))
        
        acc_train_ch = clf1_model.score(x_train_ch.reshape(-1,1),labels_train_ch)
        acc_test_ch = metrics.accuracy_score(labels_test_ch, labels_pred_ch)
        
        df_acc.loc['train_data_' + dict_ch[rx], t] = acc_train_ch
        df_acc.loc['test_data_' + dict_ch[rx], t] = acc_test_ch 
        
    df_acc.loc['5-fold CV', t] = clf1_cv_score
    
    return df_acc, clf1_model

def test_tolerance_factor_Ch(t: str, train_df: pd.DataFrame, test_df: pd.DataFrame, tolerance_factor_dict: dict[str, Any], df_acc: pd.DataFrame = pd.DataFrame(),
                          n_tresh: int = 1, cl_w: str = 'balanced', crit: str = 'entropy') -> tuple[pd.DataFrame, Any]:
    """Test a tolerance factor with sample weights for chalcogenide compounds.

    Similar to test_tolerance_factor but applies 2x weight to chalcogenide
    (S, Se) compounds during training and uses F1 score for evaluation,
    improving performance on the target chalcogenide perovskite class.

    Args:
        t: Name of tolerance factor column to test.
        train_df: Training DataFrame with tolerance factor and exp_label.
        test_df: Test DataFrame with tolerance factor and exp_label.
        tolerance_factor_dict: Dictionary to store threshold values.
        df_acc: DataFrame to accumulate F1 results across factors.
        n_tresh: Decision tree depth (1 for single threshold, 2 for range).
        cl_w: Class weight strategy for decision tree.
        crit: Split criterion ('entropy' or 'gini').

    Returns:
        tuple: Contains:
            - df_acc (pd.DataFrame): Updated F1 score results DataFrame.
            - clf1_model: Fitted decision tree classifier with sample weights.
    """
    
    
    
    labels_train=train_df["exp_label"].to_numpy()
    labels_test=test_df["exp_label"].to_numpy()

    sample_weights = train_df['rX'].to_numpy()
    sample_weights = np.where(sample_weights == 198, 2, sample_weights)
    sample_weights = np.where(sample_weights == 184, 2, sample_weights)
    sample_weights = np.where(sample_weights != 2, 1, sample_weights)
    
    if len(df_acc.columns) == 0:
        df_acc = pd.DataFrame(columns=['t_sisso', 't', 'tau', 't_jess'], index=['train_data', 'test_data', 'all_data'] )
        
    x_train_=train_df[t].to_numpy()
    x_test_=test_df[t].to_numpy()
    
    
    clf1_tree = tree.DecisionTreeClassifier(max_depth=n_tresh, class_weight=cl_w, criterion=crit)
    
    clf1_cv = cross_validate(clf1_tree, x_train_.reshape(-1,1), labels_train, scoring='f1')
    
    clf1_cv_score = np.mean(clf1_cv['test_score'])
    
    clf1_model = clf1_tree.fit(x_train_.reshape(-1,1),labels_train, sample_weight=sample_weights)
    labels_pred_=clf1_model.predict(x_test_.reshape(-1,1))
    tree.plot_tree(clf1_model)

    name_figure = 'Ch_tree_' + t + '.png'
    
    plt.savefig(TREES_DIR / name_figure)
    
    acc_train = clf1_model.score(x_train_.reshape(-1,1),labels_train)
    acc_test = metrics.accuracy_score(labels_test, labels_pred_)
    
    #General accuracy
    all_data = np.append(x_train_, x_test_)
    all_labels = np.append(labels_train, labels_test)
    labels_all_data = clf1_model.predict(all_data.reshape(-1,1))
    acc_all = metrics.accuracy_score(all_labels, labels_all_data)
    
    print('Classification tree f1 (for ' + t + ') on the train set: %f.' % acc_train)
    print('Classification tree f1 (for ' + t + ') on the train set (5 fold CV): %f.' % clf1_cv_score)
    print('Classification tree f1 (for ' + t + ') on the test set: %f.' % acc_test)

    if n_tresh == 2:
        threshold_=[clf1_model.tree_.threshold[0],clf1_model.tree_.threshold[4]]

        # add t threshold to dictionary
        tolerance_factor_dict[t].append(threshold_)

        print('%f < ' % tolerance_factor_dict[t][1][0] +  t +' < %f indicates stable perovskites.' % tolerance_factor_dict[t][1][1])
    elif n_tresh == 1:
        threshold_= clf1_model.tree_.threshold[0]

        #Add threshold to the dictionary
        tolerance_factor_dict[t].append(threshold_)

        print(t + ' < %f indicates stable perovskites.' % tolerance_factor_dict[t][1])
              
    df_acc.loc['train_data', t] = acc_train
    df_acc.loc['test_data', t] = acc_test
    df_acc.loc['all_data', t] = acc_all
    
    #get accuracy per X anion with the rX
    dict_ch = RADII_TO_ANION
    
        
    for rx in train_df.rX.unique():

        x_test_ch = test_df.loc[test_df.rX == rx, t].to_numpy()
        x_train_ch = train_df.loc[train_df.rX == rx, t].to_numpy()
        
        labels_test_ch = labels_test[test_df.rX == rx]
        labels_train_ch = labels_train[train_df.rX == rx]
        
        labels_pred_ch=clf1_model.predict(x_test_ch.reshape(-1,1))
        
        acc_train_ch = clf1_model.score(x_train_ch.reshape(-1,1),labels_train_ch)
        acc_test_ch = metrics.accuracy_score(labels_test_ch, labels_pred_ch)
        
        
        df_acc.loc['train_data_' + dict_ch[rx], t] = acc_train_ch
        df_acc.loc['test_data_' + dict_ch[rx], t] = acc_test_ch 
        
    df_acc.loc['5-fold CV', t] = clf1_cv_score
    
    return df_acc, clf1_model



if __name__ == "__main__":
    app()
