from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from loguru import logger
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np

from tf_chpvk_pv.config import FIGURES_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, RESULTS_DIR, INTERIM_DATA_DIR

app = typer.Typer()


@app.command()
def main():
    """CLI entry point for generating tolerance factor visualization plots.

    Generates Platt scaling probability plots and tolerance factor comparison
    plots (t_sisso vs t, t_jess, tau) for both raw values and calibrated
    probabilities.
    """

    platt_scaling_plot()

    plot_t_sisso_tf("t")
    plot_t_sisso_tf("t_jess")
    plot_t_sisso_tf("tau")

    plot_p_t_sisso_tf("t")
    plot_p_t_sisso_tf("t_jess")
    plot_p_t_sisso_tf("tau")


def platt_scaling_plot(t: str = 't_sisso', train_input_path: Path = RESULTS_DIR / "processed_chpvk_train_dataset.csv",
                        test_input_path: Path = RESULTS_DIR / "processed_chpvk_test_dataset.csv",
                        concat_input_path: Path = RESULTS_DIR / "processed_chpvk_concat_dataset.csv",
                        tolerance_dict_path: Path = INTERIM_DATA_DIR / "tolerance_factors.pkl",
                        output_path: Path = FIGURES_DIR / "platt_scaling_plot.png") -> None:
    """Generate scatter plot of tolerance factor vs calibrated probability.

    Creates a visualization showing the relationship between a tolerance factor
    (t_sisso by default) and its Platt-scaled probability, with points colored
    by experimental stability label and shaped by train/test split.

    Args:
        t: Name of the tolerance factor column to plot.
        train_input_path: Path to processed training dataset CSV.
        test_input_path: Path to processed test dataset CSV.
        concat_input_path: Path to save/load concatenated dataset.
        tolerance_dict_path: Path to pickle file with tolerance factor thresholds.
        output_path: Path to save the output PNG figure.
    """
    
    train_df = pd.read_csv(train_input_path)
    test_df = pd.read_csv(test_input_path)

    with open(tolerance_dict_path, 'rb') as file:
        tolerance_factor_dict = pickle.load(file)

    threshold_t_sisso = tolerance_factor_dict[t][1]

    

    if concat_input_path.exists():
        concat = pd.read_csv(concat_input_path)
    else:
        concat = pd.concat([train_df.assign(dataset='train'), test_df.assign(dataset='test')])
        concat.to_csv(concat_input_path)
    
    if 'p_' + t in train_df.columns and 'p_' + t in test_df.columns:
        logger.info("Generating Platt Scaling plot from data...")
        plt.figure(figsize=(8,8))
        plot1=sns.scatterplot(x=t, y='p_' + t, data=concat,hue='exp_label', style='dataset',
                    palette=['red','blue'], markers=['s','o'], s=80)
        
        #Set x lims
        import numpy as np
        x_lims = [min([np.min(concat[t])-0.1,threshold_t_sisso-0.5]), max([np.max(concat[t])+0.1,threshold_t_sisso+0.5])]

        plot1.set_xlabel("$t_{sisso}$", fontsize=20)
        plot1.set_ylabel("$P(t_{sisso})$", fontsize=20)
        plot1.tick_params(labelsize=20)
        plt.xlim(x_lims[0], x_lims[1])
        plt.axvline(tolerance_factor_dict[t][1])
        plt.axhline(0.5,linestyle='--')
        plt.savefig(output_path, dpi=600)

        logger.success("Plot generation complete.")

    else:
        logger.error("Platt Scaling plot cannot be generated as the required columns are not present in the dataframes.")



def platt_scaling_plot_plotly(t: str = 't_sisso', 
                        train_input_path: Path = RESULTS_DIR / "processed_chpvk_train_dataset.csv",
                        test_input_path: Path = RESULTS_DIR / "processed_chpvk_test_dataset.csv",
                        concat_input_path: Path = RESULTS_DIR / "processed_chpvk_concat_dataset.csv",
                        tolerance_dict_path: Path = INTERIM_DATA_DIR / "tolerance_factors.pkl",
                        output_path: Path = FIGURES_DIR / "platt_scaling_plot_plotly.html") -> None:  # Changed to .html for interactivity
    """Generate interactive Platt scaling plot using Plotly.

    Creates an interactive HTML visualization of tolerance factor vs calibrated
    probability with hover data, allowing exploration of individual data points.

    Args:
        t: Name of the tolerance factor column to plot.
        train_input_path: Path to processed training dataset CSV.
        test_input_path: Path to processed test dataset CSV.
        concat_input_path: Path to save/load concatenated dataset.
        tolerance_dict_path: Path to pickle file with tolerance factor thresholds.
        output_path: Path to save the interactive HTML figure.
    """
    
    import plotly.express as px
    import plotly.graph_objects as go
    
    # Load train and test datasets
    train_df = pd.read_csv(train_input_path)
    test_df = pd.read_csv(test_input_path)

    # Load tolerance factor dictionary
    with open(tolerance_dict_path, 'rb') as file:
        tolerance_factor_dict = pickle.load(file)

    threshold_t_sisso = tolerance_factor_dict[t][1]

    # Combine datasets
    if concat_input_path.exists():
        concat = pd.read_csv(concat_input_path)
    else:
        concat = pd.concat([train_df.assign(dataset='train'), test_df.assign(dataset='test')])
        concat.to_csv(concat_input_path, index=False)

    # Check if required columns exist
    if 'p_' + t in train_df.columns and 'p_' + t in test_df.columns:
        logger.info("Generating Platt Scaling plot from data...")

        # Create scatter plot using Plotly
        fig = px.scatter(
            concat, x=t, y='p_' + t, 
            color='exp_label', symbol='dataset',
            color_discrete_map={'Stable': 'red', 'Unstable': 'blue'},  # Adjust colors
            symbol_map={'train': 'square', 'test': 'circle'},  # Adjust markers
            size_max=10, 
            labels={t: r'$t_\text{sisso}$', 'p_' + t: r'$P(t_\text{sisso})$'},
            hover_data=['dataset']
        )

        # Add threshold vertical line
        fig.add_shape(
            type="line", x0=threshold_t_sisso, x1=threshold_t_sisso, y0=0, y1=1,
            line=dict(color="black", width=2, dash="dash")
        )

        # Add horizontal line at y=0.5
        fig.add_shape(
            type="line", x0=min(concat[t]), x1=max(concat[t]), y0=0.5, y1=0.5,
            line=dict(color="gray", width=2, dash="dot")
        )

        # Update layout
        fig.update_layout(
            title="Platt Scaling Plot",
            xaxis_title=r'$t_\text{sisso}$',
            yaxis_title='$P(t_\text{sisso})$',
            template="plotly_white",
            legend_title="Label"
        )

        # Save plot
        fig.write_html(output_path)  # Saves as interactive HTML
        logger.success(f"Plot saved as {output_path}")

    else:
        logger.error("Platt Scaling plot cannot be generated as the required columns are not present in the dataframes.")



def plot_t_sisso_tf(tf: str, train_input_path: Path = RESULTS_DIR / "processed_chpvk_train_dataset.csv",
                        test_input_path: Path = RESULTS_DIR / "processed_chpvk_test_dataset.csv",
                        concat_input_path: Path = RESULTS_DIR / "processed_chpvk_concat_dataset.csv",
                        tolerance_dict_path: Path = INTERIM_DATA_DIR / "tolerance_factors.pkl",
) -> None:
    """Plot t_sisso as a function of another tolerance factor.

    Creates a scatter plot comparing t_sisso against a reference tolerance
    factor (t, t_jess, or tau) to visualize their correlation, with threshold
    lines for stability regions.

    Args:
        tf: Reference tolerance factor to plot on x-axis ('t', 't_jess', or 'tau').
        train_input_path: Path to processed training dataset CSV.
        test_input_path: Path to processed test dataset CSV.
        concat_input_path: Path to save/load concatenated dataset.
        tolerance_dict_path: Path to pickle file with tolerance factor thresholds.
    """
    
    train_df = pd.read_csv(train_input_path)
    test_df = pd.read_csv(test_input_path)

    with open(tolerance_dict_path, 'rb') as file:
        tolerance_factor_dict = pickle.load(file)

    threshold_t_sisso = tolerance_factor_dict["t_sisso"][1]

    if concat_input_path.exists():
        concat = pd.read_csv(concat_input_path)
    else:
        concat = pd.concat([train_df.assign(dataset='train'), test_df.assign(dataset='test')])
        concat.to_csv(concat_input_path)

    dict_labels = {"t": "$t$", "t_sisso": "$t_{sisso}$", 't_jess': "$t_{jess}$", 'tau': r"$\tau$"}

    
    if "t_sisso" in concat.columns and tf in concat.columns:
        logger.info(f"Generating t_sisso as a function of {tf} plot from data...")
        plt.figure(figsize=(8,8))
        plot2=sns.scatterplot(x=tf, y='t_sisso', data=concat, hue='exp_label', style='dataset', 
                            palette=['red','blue'], markers=['s','o'], s=80)
        plot2.set_xlabel(dict_labels[tf], fontsize=20)
        plot2.set_ylabel(dict_labels['t_sisso'], fontsize=20)
        plot2.tick_params(labelsize=20)
        plt.ylim(threshold_t_sisso-4, threshold_t_sisso+4)
        tresholds = tolerance_factor_dict[tf][1]
        if isinstance(tresholds, list):
            plt.axvline(tresholds[0])
            plt.axvline(tresholds[1])
        else:
            plt.axvline(tresholds)
            plt.xlim(tresholds-4, tresholds+4)

        plt.axhline(threshold_t_sisso)
        name_figure = f"t_sisso as a function of {tf}.png"
        path_figure: Path = FIGURES_DIR / name_figure
        plt.savefig(path_figure)
        logger.success("Plot generation complete.")
    else:
        logger.error(f"t_sisso as a function of {tf} plot cannot be generated as the required columns are not present in the dataframes.")

def plot_p_t_sisso_tf(tf: str, train_input_path: Path = RESULTS_DIR / "processed_chpvk_train_dataset.csv",
                        test_input_path: Path = RESULTS_DIR / "processed_chpvk_test_dataset.csv",
                        concat_input_path: Path = RESULTS_DIR / "processed_chpvk_concat_dataset.csv",
                        tolerance_dict_path: Path = INTERIM_DATA_DIR / "tolerance_factors.pkl",
) -> None:
    """Plot calibrated probability P(t_sisso) as a function of another tolerance factor.

    Creates a scatter plot showing how the Platt-scaled stability probability
    varies with a reference tolerance factor, useful for comparing predictive
    power across different tolerance factor formulations.

    Args:
        tf: Reference tolerance factor to plot on x-axis ('t', 't_jess', or 'tau').
        train_input_path: Path to processed training dataset CSV.
        test_input_path: Path to processed test dataset CSV.
        concat_input_path: Path to save/load concatenated dataset.
        tolerance_dict_path: Path to pickle file with tolerance factor thresholds.
    """
    
    train_df = pd.read_csv(train_input_path)
    test_df = pd.read_csv(test_input_path)

    with open(tolerance_dict_path, 'rb') as file:
        tolerance_factor_dict = pickle.load(file)

    if concat_input_path.exists():
        concat = pd.read_csv(concat_input_path)
    else:
        concat = pd.concat([train_df.assign(dataset='train'), test_df.assign(dataset='test')])
        concat.to_csv(concat_input_path)

    dict_labels = {"t": "$t$", "p_t_sisso": "$P(t_{sisso})$", 't_jess': "$t_{jess}$", 'tau': r"$\tau$"}

    
    if "p_t_sisso" in concat.columns and tf in concat.columns:
        logger.info(f"Generating P(t_sisso) as a function of {tf} plot from data...")
        plt.figure(figsize=(8,8))
        plot2=sns.scatterplot(x=tf, y='p_t_sisso', data=concat, hue='exp_label', style='dataset', 
                            palette=['red','blue'], markers=['s','o'], s=80)
        plot2.set_xlabel(dict_labels[tf], fontsize=20)
        plot2.set_ylabel(dict_labels['p_t_sisso'], fontsize=20)
        plot2.tick_params(labelsize=20)
        tresholds = tolerance_factor_dict[tf][1]
        if isinstance(tresholds, list):
            plt.axvline(tresholds[0])
            plt.axvline(tresholds[1])
        else:
            plt.axvline(tresholds)
            plt.xlim(tresholds-4, tresholds+4)

        name_figure = f"P(t_sisso) as a function of {tf}.png"
        path_figure: Path = FIGURES_DIR / name_figure
        plt.savefig(path_figure)
        logger.success("Plot generation complete.")
    else:
        logger.error(f"P(t_sisso) as a function of {tf} plot cannot be generated as the required columns are not present in the dataframes.")


def graph_periodic_table(stable_candidates_t_sisso: List[str], t: str = 't_sisso', save_plot: bool = True, cmap_: str = 'turbo') -> None:
    """Generate periodic table heatmap showing element frequency in stable candidates.

    Creates a heatmap visualization of the periodic table where each element's
    color intensity represents how frequently it appears in compositions
    predicted to be stable perovskites.

    Args:
        stable_candidates_t_sisso: List of chemical formulas predicted as stable.
        t: Name of tolerance factor used for labeling the output file.
        save_plot: If True, save the figure to the figures directory.
        cmap_: Matplotlib colormap name for the heatmap.
    """
    from pymatviz import count_elements,  ptable_heatmap_plotly, ptable_heatmap
    import matplotlib.pyplot as plt
    import re

    element_counts = count_elements([re.sub(r'\d+', '', x) for x in stable_candidates_t_sisso])

    # Plot the periodic table heatmap
    ptable_heatmap(element_counts, log=False, heat_mode='value', cmap=cmap_)#, show_values=True)
    #fig.update_layout(title=dict(text="<b>Elements in the chemical space</b>", x=0.36, y=0.9))
    if save_plot:
      txt_title = "periodic_table_heatmap_" + t + ".png"
      plt.savefig(FIGURES_DIR / txt_title, dpi=600)

    plt.show()


def spider_plot(df: pd.DataFrame, title: str) -> None:
    """Create radar/spider plot comparing metrics across S, Se, and halide groups.

    Generates a polar plot showing normalized metric values for sulfide,
    selenide, and halide perovskite compounds, enabling visual comparison
    of multi-dimensional performance characteristics.

    Args:
        df: DataFrame indexed by group ('S', 'Se', 'hal') with metric columns.
        title: Title string used for saving the figure file.
    """

    # Libraries
    import matplotlib.pyplot as plt
    import pandas as pd
    from math import pi


    # ------- PART 1: Create background

    # number of variable
    categories=list(df.drop(columns=['group']))
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    plt.figure(figsize=(15,15))
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2,0.4,0.6, 0.8], ["0.2","0.4","0.6", "0.8"], color="grey", size=20)
    plt.ylim(0,1)


    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    # I don't make a loop, because plotting more than 3 groups makes the chart unreadable

    # Ind1
    values=df.loc['S'].drop(['group']).values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="S")
    ax.fill(angles, values, 'b', alpha=0.1)

    # Ind2
    values=df.loc['Se'].drop(['group']).values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="Se")
    ax.fill(angles, values, 'r', alpha=0.1)

    # Ind3
    values=df.loc['hal'].drop(['group']).values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="Hal.")
    ax.fill(angles, values, 'r', alpha=0.1)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), prop={'size': 35})
    
    #change font_size
    #ax.tick_params(labelsize=5)
    
    #save the graph

    txt_title = 'radar plot - ' + title + '.png'

    plt.savefig(FIGURES_DIR / txt_title, bbox_inches='tight')


def plot_tau_star_histogram(threshold: float, df: pd.DataFrame) -> None:
    """Create histogram of tau* tolerance factor values by stability class.

    Generates a histogram showing the distribution of tau* (τ*) values,
    colored by perovskite/nonperovskite classification, with the stability
    threshold region highlighted.

    Args:
        threshold: Tau* threshold value for stable perovskite classification.
        df: DataFrame containing 'tau*' and 'exp_label' columns.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig = plt.figure(figsize=(8, 6))

    df_ = df.copy()
    df_.loc[df_['exp_label'] == 1, 'exp_label_'] = 'Perovskite'
    df_.loc[df_['exp_label'] == 0, 'exp_label_'] = 'Nonperovskite'


    sns.set_context('talk')
    ax = sns.histplot(data=df_, x='tau*', hue='exp_label_',
                      multiple='dodge', element='bars', bins=25,
                      hue_order=['Perovskite', 'Nonperovskite'])

    # Add axvspan calls with labels
    ax.axvline(x=threshold, color='k', label='$\\tau$*' + f' > {threshold}')
    ax.axvspan(xmin=0, xmax=threshold, color='limegreen', alpha=0.25, label= '$\\tau$*' + f' < {threshold}')

    # Get all handles and labels from the axis. This should include both histplot and axvspan.
    if ax.legend_ is not None:
        ax.legend_.set_title(None)

    # Create a unified legend from the collected handles and labels, without a title.
    # This will overwrite any default legend created by seaborn.
    #ax.legend(handles=handles, labels=labels, title=None)

    plt.xlim([0, 1.6])
    plt.xlabel('SISSO-derived $\\tau$*')
    plt.ylabel('Counts')
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / 'tau_star_histogram.png', dpi=600, bbox_inches='tight')

    plt.show()


def plot_tau_star_histogram_interactive(threshold: float, df: pd.DataFrame) -> Any:
    """Create interactive histogram of tau* tolerance factor values by stability class.

    Interactive Plotly version of :func:`plot_tau_star_histogram`. Generates a
    grouped bar histogram showing the distribution of tau* (τ*) values colored
    by perovskite/nonperovskite classification, with the stability threshold
    region highlighted.

    Args:
        threshold: Tau* threshold value for stable perovskite classification.
        df: DataFrame containing 'tau*' and 'exp_label' columns.

    Returns:
        plotly.graph_objects.Figure: Interactive histogram figure.
    """
    import plotly.graph_objects as go

    df_ = df.copy()
    df_['exp_label_'] = df_['exp_label'].map({1: 'Perovskite', 0: 'Nonperovskite'})

    colors = {'Perovskite': '#1f77b4', 'Nonperovskite': '#ff7f0e'}

    fig = go.Figure()

    for label in ['Perovskite', 'Nonperovskite']:
        subset = df_[df_['exp_label_'] == label]
        fig.add_trace(go.Histogram(
            x=subset['tau*'],
            name=label,
            nbinsx=25,
            marker_color=colors[label],
            opacity=0.8,
        ))

    # Green shaded region [0, threshold]
    fig.add_shape(
        type='rect',
        xref='x', yref='paper',
        x0=0, x1=threshold,
        y0=0, y1=1,
        fillcolor='limegreen',
        opacity=0.25,
        line_width=0,
        layer='below',
    )

    # Vertical threshold line
    fig.add_shape(
        type='line',
        xref='x', yref='paper',
        x0=threshold, x1=threshold,
        y0=0, y1=1,
        line=dict(color='black', width=2),
    )

    fig.update_layout(
        barmode='group',
        xaxis=dict(title='SISSO-derived τ*', range=[0, 1.6]),
        yaxis=dict(title='Counts'),
        template='plotly_white',
        legend=dict(title=None),
    )

    return fig


def plot_t_star_histogram(thresholds: List[float], df: pd.DataFrame) -> None:
    """Create histogram of t* (Jess et al.) tolerance factor values by stability class.

    Generates a histogram showing the distribution of t* values with
    perovskite/nonperovskite classification, highlighting the two-threshold
    stability region characteristic of this tolerance factor.

    Args:
        thresholds: List of two threshold values [lower, upper] defining stable region.
        df: DataFrame containing 't*' and 'exp_label' columns.
    """

    fig = plt.figure(figsize=(8, 6))

    df_ = df.copy()
    df_.loc[df_['exp_label'] == 1, 'exp_label_'] = 'Perovskite'
    df_.loc[df_['exp_label'] == 0, 'exp_label_'] = 'Nonperovskite'


    sns.set_context('talk')
    ax = sns.histplot(data=df_, x='t*', hue='exp_label_',
                        multiple='dodge', element='bars', bins=25,
                        hue_order=['Perovskite', 'Nonperovskite']
                        )

    # Add axvspan calls with labels
    ax.axvline(x=thresholds[0], color='k')
    ax.axvline(x=thresholds[1], color='k')

    ax.axvspan(xmin=thresholds[0], xmax=thresholds[1], color='limegreen', alpha=0.25, )

    # Get all handles and labels from the axis. This should include both histplot and axvspan.
    if ax.legend_ is not None:
        ax.legend_.set_title(None)

    # Create a unified legend from the collected handles and labels, without a title.
    # This will overwrite any default legend created by seaborn.
    #ax.legend(handles=handles, labels=labels, title=None)

    plt.xlim([0.3, 2.2])
    plt.xlabel('Jess et al. tolerance factor ($t_{Jess}$)')
    plt.ylabel('Counts')
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / 't_star_histogram.png', dpi=600, bbox_inches='tight')

    plt.show()


def plot_t_star_histogram_interactive(thresholds: List[float], df: pd.DataFrame) -> Any:
    """Create interactive histogram of t* (Jess et al.) tolerance factor values by stability class.

    Interactive Plotly version of :func:`plot_t_star_histogram`. Generates a
    grouped bar histogram showing the distribution of t* values with
    perovskite/nonperovskite classification, highlighting the two-threshold
    stability region.

    Args:
        thresholds: List of two threshold values [lower, upper] defining stable region.
        df: DataFrame containing 't*' and 'exp_label' columns.

    Returns:
        plotly.graph_objects.Figure: Interactive histogram figure.
    """
    import plotly.graph_objects as go

    df_ = df.copy()
    df_['exp_label_'] = df_['exp_label'].map({1: 'Perovskite', 0: 'Nonperovskite'})

    colors = {'Perovskite': '#1f77b4', 'Nonperovskite': '#ff7f0e'}

    fig = go.Figure()

    for label in ['Perovskite', 'Nonperovskite']:
        subset = df_[df_['exp_label_'] == label]
        fig.add_trace(go.Histogram(
            x=subset['t*'],
            name=label,
            nbinsx=25,
            marker_color=colors[label],
            opacity=0.8,
        ))

    # Green shaded region between the two thresholds
    fig.add_shape(
        type='rect',
        xref='x', yref='paper',
        x0=thresholds[0], x1=thresholds[1],
        y0=0, y1=1,
        fillcolor='limegreen',
        opacity=0.25,
        line_width=0,
        layer='below',
    )

    # Vertical threshold lines
    for thr in thresholds:
        fig.add_shape(
            type='line',
            xref='x', yref='paper',
            x0=thr, x1=thr,
            y0=0, y1=1,
            line=dict(color='black', width=2),
        )

    fig.update_layout(
        barmode='group',
        xaxis=dict(title='Jess et al. tolerance factor (t<sub>Jess</sub>)', range=[0.3, 2.2]),
        yaxis=dict(title='Counts'),
        template='plotly_white',
        legend=dict(title=None),
    )

    return fig


def plot_t_star_vs_p_t_sisso(df: pd.DataFrame, thresholds: List[float]) -> None:
    """Create scatter plot of t* vs P(τ*) with stability regions.

    Visualizes the relationship between the Jess et al. tolerance factor (t*)
    and the calibrated probability P(τ*), with vertical lines marking the
    stability threshold region.

    Args:
        thresholds: List of two threshold values [lower, upper] for t* stability region.
        df: DataFrame containing 't*', 'p_tau*', and 'exp_label' columns.
    """
    plt.figure(figsize=(8, 6))

    # Create a copy and map exp_label for better legend labels
    df_plot = df.copy()
    df_plot['exp_label_'] = df_plot['exp_label'].map({0: 'Nonperovskite', 1: 'Perovskite'})

    markers = {"Nonperovskite": "X", "Perovskite": "o"}

    ax = sns.scatterplot(
        data=df_plot,
        x='t*',
        y='p_tau*',
        hue='exp_label_',
        style='exp_label_',
        s=100, # size of the points
        alpha=1, # transparency
        hue_order=['Perovskite', 'Nonperovskite'],
        markers=markers,
    )

    ax.axvline(x=thresholds[0], color='k')
    ax.axvline(x=thresholds[1], color='k')

    ax.axvspan(xmin=thresholds[0], xmax=thresholds[1], color='limegreen', alpha=0.25, )

    ax.set_xlabel('Jess et al. tolerance factor ($t_{Jess}$)')
    ax.set_ylabel('P($\\tau$*)')

    # Remove title from legend
    if ax.legend_ is not None:
        ax.legend_.set_title(None)
        sns.move_legend(ax, loc='upper right')

    plt.xlim([0.3, 2.2])
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / 'P_tau_t_star_scatter.png', dpi=600, bbox_inches='tight')

    plt.show()


def plot_t_star_vs_p_t_sisso_interactive(df: pd.DataFrame, thresholds: List[float]) -> Any:
    """Create interactive scatter plot of t* vs P(τ*) with stability regions.

    Interactive Plotly version of :func:`plot_t_star_vs_p_t_sisso`. Visualizes
    the relationship between the Jess et al. tolerance factor (t*) and the
    calibrated probability P(τ*), with vertical lines marking the stability
    threshold region and hover text for each compound.

    Args:
        df: DataFrame containing 't*', 'p_tau*', and 'exp_label' columns.
        thresholds: List of two threshold values [lower, upper] for t* stability region.

    Returns:
        plotly.graph_objects.Figure: Interactive scatter figure.
    """
    import plotly.graph_objects as go

    df_plot = df.copy()
    df_plot['exp_label_'] = df_plot['exp_label'].map({1: 'Perovskite', 0: 'Nonperovskite'})

    marker_styles = {
        'Perovskite':    dict(symbol='circle', color='#1f77b4', size=9, opacity=1.0,
                              line=dict(color='#1f77b4', width=1)),
        'Nonperovskite': dict(symbol='x',      color='#ff7f0e', size=9, opacity=1.0,
                              line=dict(color='#ff7f0e', width=1)),
    }

    fig = go.Figure()

    for label in ['Perovskite', 'Nonperovskite']:
        subset = df_plot[df_plot['exp_label_'] == label]
        hover = (
            subset['formula'].apply(lambda f: f'<b>{f}</b><br>')
            if 'formula' in subset.columns
            else pd.Series([''] * len(subset), index=subset.index)
        )
        hover = hover + subset['t*'].apply(lambda v: f't<sub>Jess</sub> = {v:.3f}<br>')
        hover = hover + subset['p_tau*'].apply(lambda v: f'P(τ*) = {v:.3f}')
        fig.add_trace(go.Scatter(
            x=subset['t*'],
            y=subset['p_tau*'],
            mode='markers',
            name=label,
            marker=marker_styles[label],
            hovertext=hover,
            hoverinfo='text',
        ))

    # Green shaded region between the two thresholds
    fig.add_shape(
        type='rect',
        xref='x', yref='paper',
        x0=thresholds[0], x1=thresholds[1],
        y0=0, y1=1,
        fillcolor='limegreen',
        opacity=0.25,
        line_width=0,
        layer='below',
    )

    # Vertical threshold lines
    for thr in thresholds:
        fig.add_shape(
            type='line',
            xref='x', yref='paper',
            x0=thr, x1=thr,
            y0=0, y1=1,
            line=dict(color='black', width=2),
        )

    fig.update_layout(
        xaxis=dict(title='Jess et al. tolerance factor (t<sub>Jess</sub>)', range=[0.3, 2.2]),
        yaxis=dict(title='P(τ*)'),
        template='plotly_white',
        legend=dict(title=None, x=1.0, xanchor='right', y=1.0),
    )

    return fig


def colormap_radii(df: pd.DataFrame, exp_df: pd.DataFrame, clf_proba: Optional[Any] = None, t_sisso: bool = False) -> None:
    """Create 2D colormap of stability predictions vs ionic radii for S and Se anions.

    Generates side-by-side heatmaps showing t_sisso or P(t_sisso) values across
    the (rA, rB) ionic radii space for sulfide and selenide compounds, with
    experimentally observed compounds overlaid as markers.

    Args:
        df: DataFrame with predicted compositions containing rA, rB, rX columns.
        exp_df: DataFrame with experimental compounds for overlay markers.
        clf_proba: Pre-trained Platt scaling classifier; if None, loads from file.
        t_sisso: If True, plot raw t_sisso values; if False, plot P(t_sisso).
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    if clf_proba is None:
        from tf_chpvk_pv.modeling.train import train_platt_scaling

        train_df = pd.read_csv(RESULTS_DIR / "processed_chpvk_train_dataset.csv")
        test_df = pd.read_csv(RESULTS_DIR / "processed_chpvk_test_dataset.csv")
        
        with open(INTERIM_DATA_DIR / "tolerance_factor_classifiers.pkl", 'rb') as file:
            clfs = pickle.load(file)

        train_df, test_df, clf_proba = train_platt_scaling(train_df, test_df, clfs['t_sisso'])

    rA_range = [df.rA.min()-10, df.rA.max()+10]
    rB_range = [df.rB.min()-10, df.rB.max()+10]
    rX_range = [df.rX.min(), df.rX.max()]

    def calculate_t_sisso(rA, rB, rX):
        #(|((rA_rX_ratio + rB_rX_ratio) + (|rB_rX_ratio - log_rA_rB_ratio|)) - (rA_rX_ratio**3)|)
        #return np.abs( (np.abs( (rA/rX * rB/rX) - (np.exp(rB/rX)) )) - (np.abs( np.abs(rB/rX - np.log(rA/rB)) - (rA/rX) ) ) )
        return np.abs( ( (rA/rX + rB/rX) + (np.abs(rB/rX - np.log(rA/rB)) )) - (rA/rX)**3 )

    #All the radii in pm
    rA = np.linspace(rA_range[0], rA_range[1], 1000)
    rB = np.linspace(rB_range[0], rB_range[1], 1000)

    xv, yv = np.meshgrid(rA, rB)

    t_sisso_S = calculate_t_sisso(xv, yv, rX_range[0])
    t_sisso_Se = calculate_t_sisso(xv, yv, rX_range[1])

    p_t_sisso_S = clf_proba.predict_proba(t_sisso_S.reshape(-1,1))[:,1]
    p_t_sisso_Se = clf_proba.predict_proba(t_sisso_Se.reshape(-1,1))[:,1]

    sns.set_context('talk')

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    plt.subplots_adjust(wspace=0.2) # Increased wspace

    if t_sisso:
        color_map = 'Pastel1'
    else:
        color_map = 'coolwarm_r'

    rA_range = [110, 180]
    rB_range = [50, 120]

    # Plot for S anion (rX_range[0])
    if t_sisso:
        scatter_s = axes[0].scatter(xv, yv, c=t_sisso_S, cmap=color_map, vmin=0.5, vmax=2.5)
    else:
        scatter_s = axes[0].scatter(xv, yv, c=p_t_sisso_S, cmap=color_map, vmin=0, vmax=1)
    axes[0].set_xlabel('$r_A$ (pm)')
    axes[0].set_ylabel('$r_B$ (pm)')
    axes[0].set_xlim(rA_range)
    axes[0].set_ylim(rB_range)
    axes[1].tick_params(left=True, right=True)
    axes[0].set_title('ABS$_3$ compounds')

    # Plot for Se anion (rX_range[1])
    if t_sisso:
        axes[1].scatter(xv, yv, c=t_sisso_Se, cmap=color_map, vmin=0.5, vmax=2.5)
    else:
        axes[1].scatter(xv, yv, c=p_t_sisso_Se, cmap=color_map, vmin=0, vmax=1)
    axes[1].set_xlabel('$r_A$ (pm)')
    axes[1].set_xlim(rA_range)
    axes[1].set_ylim(rB_range)
    axes[1].set_yticklabels([])
    axes[1].tick_params(left=True, right=True)
    axes[1].set_title('ABSe$_3$ compounds')

    # Add a single color bar for both subplots
    if t_sisso:
        fig.colorbar(scatter_s, ax=axes.ravel().tolist(), label='$\\tau*$')
    else:
        fig.colorbar(scatter_s, ax=axes.ravel().tolist(), label='$P(\\tau*)$')

    #remove non-chalcogenides from exp_df
    exp_df = exp_df[exp_df.rX.isin(rX_range)]
    #delete rX = 196 pm (Br) from exp_df
    exp_df = exp_df[exp_df.rX != 196]

    #Add experimentally observed compounds
    for idx in exp_df.index:
        d = exp_df.loc[idx]
        rA_ = d.rA
        rB_ = d.rB
        X = d.rX
        if d.exp_label == 1:
            if X == rX_range[0]:
                axes[0].scatter(rA_, rB_, marker='s', color='lightgray', s=50, edgecolor='black')
            elif X == rX_range[1]:
                axes[1].scatter(rA_, rB_, marker='s', color='lightgray', s=50, edgecolor='black')
        else:
            if X == rX_range[0]:
                axes[0].scatter(rA_, rB_, marker='^', color='lightgray', s=50, edgecolor='black')
            elif X == rX_range[1]:
                axes[1].scatter(rA_, rB_, marker='^', color='lightgray', s=50, edgecolor='black')

    if t_sisso:
        plt.savefig(FIGURES_DIR / 't_sisso color_map radii.png', dpi=600)
    else:
        plt.savefig(FIGURES_DIR / 'p_t_sisso color_map radii.png', dpi=600)
    plt.show()


def confusion_matrix_plot(df: pd.DataFrame, test: bool = True) -> None:
    """Generate confusion matrix heatmap for stability classification results.

    Creates a heatmap visualization of the confusion matrix comparing predicted
    stability (based on P(t_sisso) >= 0.5 threshold) against experimental labels.

    Args:
        df: DataFrame containing 'exp_label' and 'p_t_sisso' columns.
        test: If True, label as test set and use fixed color scale;
            if False, label as train set with auto-scaled colors.
    """


    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    sns.set_context('talk')

    y_true = df['exp_label'].values

    y_pred = np.zeros_like(y_true)

    y_pred[df['p_t_sisso'] >= 0.5] = 1

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 4))
    if test:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'], vmax=35, vmin=0)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if test:
        plt.title('Confusion Matrix - Test Set')
    else:
        plt.title('Confusion Matrix - Train Set')

    plt.savefig(FIGURES_DIR / f'confusion_matrix_{"test" if test else "train"}.png', dpi=600, bbox_inches='tight')
    plt.show()

def normalize_abx3(formula: str) -> Optional[str]:
    """Normalize a perovskite formula to standard ABX3 format.

    Parses a chemical formula and reorders elements to place the two cations
    (A, B) alphabetically before the anion (X) with stoichiometry 3.

    Args:
        formula: Chemical formula string (e.g., 'BaTiS3', 'SrZrSe3').

    Returns:
        str: Normalized formula in 'ABX3' format, or None if formula
            doesn't match ABX3 stoichiometry.
    """
    import re
    from collections import Counter
    tokens = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    counts = Counter({el: int(num) if num else 1 for el, num in tokens})

    # X has stoichiometry 3
    X = [el for el, c in counts.items() if c == 3]
    if len(X) != 1:
        return None  # or raise error if you prefer
    X = X[0]

    # A and B are the remaining elements
    AB = sorted(el for el in counts if el != X)
    if len(AB) != 2:
        return None

    return f"{AB[0]}{AB[1]}{X}3"

def plot_matrix(df_out: pd.DataFrame, df_crystal: pd.DataFrame, anion: str = 'S', parameter: str = 'Eg', clf_proba: Optional[Any] = None) -> None:
    """Create scatter matrix of cation A vs cation B colored by bandgap or P(t_sisso).

    Generates a matrix plot showing all predicted compositions for a given anion,
    with each point representing an AB pair colored by bandgap energy or stability
    probability. CrystaLLM-validated compositions are highlighted with borders.

    Args:
        df_out: DataFrame with predicted compositions containing A, B, X, formula columns.
        df_crystal: DataFrame with CrystaLLM-validated compositions for highlighting.
        anion: Anion element to filter ('S' or 'Se').
        parameter: Coloring parameter ('Eg' for bandgap, 'p_t_sisso' for probability).
        clf_proba: Pre-trained Platt scaling classifier; if None, loads from file.
    """


    import matplotlib.pyplot as plt
    from matplotlib.markers import MarkerStyle
    from tf_chpvk_pv.config import FIGURES_DIR

    import numpy as np
    import seaborn as sns

    element_to_number = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
    "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40,
    "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
    "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
    "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100,
    "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109,
    "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118
    }

    if parameter == 'p_t_sisso' and clf_proba is None:
        from tf_chpvk_pv.modeling.train import train_platt_scaling

        train_df = pd.read_csv(RESULTS_DIR / "processed_chpvk_train_dataset.csv")
        test_df = pd.read_csv(RESULTS_DIR / "processed_chpvk_test_dataset.csv")
        
        with open(INTERIM_DATA_DIR / "tolerance_factor_classifiers.pkl", 'rb') as file:
            clfs = pickle.load(file)

        train_df, test_df, clf_proba = train_platt_scaling(train_df, test_df, clfs['t_sisso'])
        df_out['p_t_sisso'] = clf_proba.predict_proba(df_out['t_sisso'].values.reshape(-1,1))[:,1]     

    #Normalize formulas to ABX3 format for matching
    df_out["norm_formula"] = df_out["formula"].apply(normalize_abx3)
    df_crystal["norm_formula"] = df_crystal["formula"].apply(normalize_abx3)
    df_sisso = df_out[df_out['norm_formula'].isin(df_crystal['norm_formula'])]
    
    df_out = df_out[df_out['X'] == anion].copy()
    if parameter in ['Eg']:
        df_out = df_out[df_out[parameter] > 0].copy()
    df_out['Z_A'] = df_out.A.map(element_to_number)
    df_out['Z_B'] = df_out.B.map(element_to_number)
    df_out.sort_values(by=['Z_A', 'Z_B'], inplace=True, ascending=[True, True])

    df_sisso = df_sisso[df_sisso['X'] == anion].copy()
    if parameter in ['Eg']:
        df_sisso = df_sisso[df_sisso[parameter] > 0].copy()
    df_sisso['Z_A'] = df_sisso.A.map(element_to_number)
    df_sisso['Z_B'] = df_sisso.B.map(element_to_number)
    df_sisso.sort_values(by=['Z_A', 'Z_B'], inplace=True, ascending=[True, True])

    # Build x-axis positions sorted by atomic number
    sorted_B = sorted(df_out['B'].unique(), key=lambda x: element_to_number[x])
    b_to_idx = {b: i for i, b in enumerate(sorted_B)}
    df_out['x_pos'] = df_out['B'].map(b_to_idx)
    df_sisso['x_pos'] = df_sisso['B'].map(b_to_idx)

    #sns.set_theme(style="whitegrid")
    sns.set_context('talk')

    df_crab = df_sisso

    size_markers = 200

    if parameter == 'Eg':
        if anion == 'S':
            fsize = df_out.nunique()['A'] / 2.8
        elif anion == 'Se':
            fsize = df_out.nunique()['A'] / 2.6
    elif parameter == 'p_t_sisso':
        fsize = df_out.nunique()['A'] / 3

    fig, ax = plt.subplots(figsize=(fsize+3, fsize))


    

    if parameter == 'Eg':
        vmin_, vmax_ = 0.5, 3.5
        cmap_ = 'jet'
    elif parameter == 'p_t_sisso':
        vmin_, vmax_ = 0.0, 1.0
        cmap_ = 'coolwarm_r'
    else:
        vmin_, vmax_ = df_out[parameter].min(), df_out[parameter].max()

        
    im1 = ax.scatter(df_out.x_pos, df_out.A, c=df_out[parameter], marker='s',
                vmin=vmin_, vmax=vmax_, cmap=cmap_, s=size_markers)
    
    ax.scatter(df_crab.x_pos, df_crab.A, marker='s',
                edgecolor="black", color="None", s=size_markers)

    cbar1 = plt.colorbar(im1, ax=ax)
    if parameter == 'Eg':
        cbar1.set_label("Bandgap (eV) ", rotation=90)
    elif parameter == 'p_t_sisso':
        cbar1.set_label(r"$P(\tau*)$ ", rotation=90)


    plt.title('')

    # Tweak the figure to finalize
    ax.set(xlabel="Cation B", ylabel="Cation A", aspect="equal")

    ax.set_xticks(range(len(sorted_B)))
    ax.set_xticklabels(sorted_B, rotation=90)
    #plt.grid(color='None', linestyle='-')
    #ax.set_frame_on(False)
    ax.grid(False)

    if parameter == 'Eg':
        name_file = 'matrix_' + f'{parameter.lower()}_crabnet_' + anion + '.png'
    elif parameter == 'p_t_sisso':
        name_file = 'matrix_' + f'{parameter.lower()}_' + anion + '.png'

    plt.savefig(FIGURES_DIR / name_file, dpi=600, bbox_inches='tight')

    plt.show()

def pareto_front_plot(df: pd.DataFrame, variable: str, Eg_ref: float = 1.34,
                  plot_names: bool = False, ax: Optional[Any] = None,
                  same_y_axis: bool = False, 
                  plot_PCE: bool = False,
                  sj_limit_path: Path = RAW_DATA_DIR / "SJ_limit.csv",
                  dj_limit_path: Path = RAW_DATA_DIR / "DJ_limit.csv") -> None:
    """Plot Pareto front for bandgap deviation vs sustainability metric.

    Creates a multi-objective optimization plot showing the tradeoff between
    bandgap deviation from a reference value and a sustainability metric
    (HHI, SR, or 1-CL score), with Pareto-optimal points highlighted.

    Args:
        df: DataFrame with 'bandgap', 'formula', and sustainability metric columns.
        variable: Sustainability metric column name ('HHI', 'SR', or '1-CL score').
        Eg_ref: Reference bandgap in eV (1.34 for single-junction, 1.71 for tandem).
        plot_names: If True, annotate points with formula labels.
        ax: Matplotlib axes object; if None, creates new figure.
        same_y_axis: If True, hide y-axis labels (for multi-panel figures).
        plot_PCE: If True, overlay theoretical PCE limit colormap.
        sj_limit_path: Path to single-junction Shockley-Queisser limit data.
        dj_limit_path: Path to dual-junction limit data for tandem cells.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    sns.set_context('talk')

    df = df[df['B'] != 'U']
    df = df[df['A'] != 'U'].reset_index(drop=True)
    cols = [variable, 'Eg_dev']
    df["Eg_dev"] = abs(Eg_ref - df["bandgap"])
    is_pareto = np.ones(len(df), dtype=bool)
    for i, point in df[cols].iterrows():
        if is_pareto[i]:
            is_pareto[is_pareto] = ~(
                (df.loc[is_pareto, cols] >= point).all(axis=1) &
                (df.loc[is_pareto, cols] > point).any(axis=1)
            )
            is_pareto[i] = True

    df_pareto = df[is_pareto]
    
    if ax is None:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
    

    #set_scatter_colors
    df['color'] = 'black'
    df.loc[df['Eg_dev'] / Eg_ref <= 0.10, 'color'] = 'blue'
    df.loc[( df[variable]-df[variable].min() ) / (df[variable].max() - df[variable].min()) <= 0.10, 'color'] = 'blue'
    df.loc[is_pareto, 'color'] = 'red'

    df_black = df[df['color'] == 'black']
    df_blue = df[df['color'] == 'blue']
    df_red = df[df['color'] == 'red']

    ax.scatter(df_black["Eg_dev"], df_black[variable], color='black',
               edgecolor='k', marker='o', alpha=0.7)

    ax.scatter(df_blue["Eg_dev"], df_blue[variable], color='blue',
               edgecolor='k', marker='s', alpha=0.7)

    ax.scatter(df_red["Eg_dev"], df_red[variable], color='red',
               edgecolor='k', marker='^')
    
    xlims = [-0.05, max(df["Eg_dev"])*1.1]

    if '1-CL' in variable:
        ylims = [0, max(df[variable])*1.1]
    else:
        ylims = [0.1, max(df[variable])*1.1]

    if plot_PCE:
        # Load Shockley-Queisser limit data
        sj_limit = pd.read_csv(sj_limit_path)
        dj_limit = pd.read_csv(dj_limit_path)

        if Eg_ref == 1.34:
            pcolor = ax.pcolorfast(xlims, ylims, sj_limit[ 'PCE (%)'].values.reshape(1, -1),
                                   vmin=0, vmax=35,
                                   cmap='BuGn', alpha=0.45, label='SJ Limit (1.34 eV)')
            cbar = plt.colorbar(pcolor, ax=ax, label='PCE theoretical limit (%)', location='top')
            cbar.ax.invert_xaxis()
        elif Eg_ref == 1.71:
            pcolor = ax.pcolorfast(xlims, ylims, dj_limit[ 'PCE (%)'].values.reshape(1, -1),
                                   vmin=29, vmax=45,
                                   cmap='PuBu', alpha=0.45, label='DJ Limit (1.71 eV)')
            cbar = plt.colorbar(pcolor, ax=ax, label='PCE theoretical limit (%)', location='top')
            cbar.ax.invert_xaxis()
        
    
    
    if plot_names:
        for _, row in df.iterrows():
            ax.text(row["Eg_dev"], row[variable], row["formula"].replace("3", "$_3$"), fontdict={'fontsize':8})
    
    variable_title = {'HHI': 'Herfindahl-Hirschman Index (HHI)',
                       'SR': 'Supply Risk (SR)',
                       '1-CL score': '1 - Crystal-likeness Score (CLS)',}

    ax.set_xlabel("|{0:0.2f} - $E_g$| (eV)".format(Eg_ref))
    if not same_y_axis:
        ax.set_ylabel(variable_title[variable])
    else:
        ax.set_ylabel(None)
        ax.set_yticklabels([])
    
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    if ax is None:
        plt.show()

def plot_pareto_3fronts(df: pd.DataFrame, print_tables: bool = False, plot_names: bool = False, FIGURES_DIR: Path = FIGURES_DIR) -> None:
    """Create 3-objective Pareto front plot for bandgap, SR, and CL score.

    Visualizes the multi-objective optimization landscape with crystal-likeness
    score vs supply risk, colored by bandgap. Highlights Pareto-optimal
    candidates for both single-junction (1.34 eV) and tandem (1.71 eV) targets.

    Args:
        df: DataFrame with 'bandgap', 'SR', 'CL score', and 'formula' columns.
        print_tables: If True, print DataFrames of Pareto-optimal compositions.
        plot_names: If True, annotate all points with formula labels.
        FIGURES_DIR: Directory path for saving output figures.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_context('talk')

    def pareto_front_3obj(df, objectives):

        import numpy as np

        is_pareto = np.ones(len(df), dtype=bool)
        for i, point in df[objectives].iterrows():
            if is_pareto[i]:
                is_pareto[is_pareto] = ~(
                    (df.loc[is_pareto, objectives] >= point).all(axis=1) &
                    (df.loc[is_pareto, objectives] > point).any(axis=1)
                )
                is_pareto[i] = True
        return df[is_pareto]
    

    df = df[df['B'] != 'U']
    df = df[df['A'] != 'U'].reset_index(drop=True)

    df['Eg_dev_SJ'] = abs(df['bandgap'] - 1.34)
    df['Eg_dev_T'] = abs(df['bandgap'] - 1.71)
    df['1-CL score'] = 1 - df['CL score']

    objectives = ['Eg_dev_SJ', 'SR', '1-CL score']
    df_pareto_SJ = pareto_front_3obj(df, objectives)

    objectives = ['Eg_dev_T', 'SR', '1-CL score']
    df_pareto_T = pareto_front_3obj(df, objectives)

    if print_tables:
        print("Pareto front for Single Junction bandgap (1.34 eV):")
        print(df_pareto_SJ[['formula', 'bandgap', 'bandgap_sigma', 'SR', 'CL score', 'CL score std']])
        print("\nPareto front for Tandem configuration bandgap (1.71 eV):")
        print(df_pareto_T[['formula', 'bandgap', 'bandgap_sigma', 'SR', 'CL score', 'CL score std']])

    fig, ax = plt.subplots(figsize=(8, 6))
    df_ = df[~df['formula'].isin(df_pareto_SJ['formula'].tolist() + df_pareto_T['formula'].tolist())]
    sc = ax.scatter(df_["CL score"], df_["SR"], c=df_["bandgap"], cmap="jet", vmin=0.5, vmax=3.5, s=100, edgecolors='darkgray')
    df_pareto_SJ_ = df_pareto_SJ[df_pareto_SJ['bandgap'] < 1.51]
    ax.scatter(df_pareto_SJ_["CL score"], df_pareto_SJ_["SR"], c=df_pareto_SJ_["bandgap"], cmap="jet", vmin=0.5, vmax=3.5, marker='^', edgecolors='black', s=100)
    df_pareto_T_ = df_pareto_T[df_pareto_T['bandgap'] >= 1.51]
    ax.scatter(df_pareto_T_["CL score"], df_pareto_T_["SR"], c=df_pareto_T_["bandgap"], cmap="jet", vmin=0.5, vmax=3.5, marker='s', edgecolors='black', s=100)
    
    
    plt.colorbar(sc, label="Bandgap (eV)")
    ax.set_xlabel("Crystal-likeness Score (CLS)")
    ax.set_ylabel("Supply Risk (SR)")

    if plot_names:
        for _, row in df_.iterrows():
            ax.text(row["CL score"], row["SR"], row["formula"].replace("3", "$_3$"), fontdict={'fontsize':8})
        for _, row in df_pareto_SJ.iterrows():
            ax.text(row["CL score"], row["SR"], row["formula"].replace("3", "$_3$"))
        for _, row in df_pareto_T.iterrows():
            ax.text(row["CL score"], row["SR"], row["formula"].replace("3", "$_3$"))
        fig.savefig(FIGURES_DIR / "pareto_front_CL_score_SR_Eg_with_names.png", dpi=300, bbox_inches='tight')
    else:
        fig.savefig(FIGURES_DIR / "pareto_front_CL_score_SR_Eg.png", dpi=300, bbox_inches='tight')
    
    plt.show()  


def plot_PCA(df_scaled: pd.DataFrame, df_pca: pd.DataFrame, original_df: pd.DataFrame, component_loadings: pd.DataFrame, pca: Any, pc1: int = 1, pc2: int = 2) -> None:
    """Create biplot of PCA scores with feature loading vectors.

    Generates a visualization combining PCA-transformed data points (colored by
    bandgap, sized by rB) with feature loading vectors showing how original
    variables contribute to the principal components.

    Args:
        df_scaled: Standardized DataFrame from perform_pca().
        df_pca: Original data subset used for PCA.
        original_df: Full original DataFrame with 'color_edge' column for styling.
        component_loadings: DataFrame of feature loadings from perform_pca().
        pca: Fitted sklearn PCA object.
        pc1: Principal component number for x-axis (1-indexed).
        pc2: Principal component number for y-axis (1-indexed).
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from tf_chpvk_pv.config import FIGURES_DIR

    dict_values = {'nA':'$n_A$',
                  'nB':'$n_B$',
                  'nX':'$n_X$',
                  'chi_A':r'$\chi_A$',
                  'chi_B':r'$\chi_B$',
                  'chi_X':r'$\chi_X$',
                  'rX':'$r_X$',
                  'rA':'$r_A$',
                  'rB':'$r_B$',
                  'bandgap': '$E_g$'}

    sns.set_context('talk')

    # Transform the scaled data to get the principal components scores
    pca_scores = pca.transform(df_scaled)

    # Get the loadings for PC1 and PC2
    loadings_pc1 = component_loadings['PC' + str(pc1)]
    loadings_pc2 = component_loadings['PC' + str(pc2)]

    # Scale the loadings for better visualization if needed (e.g., by explained variance or just a constant factor)
    # For simplicity, we'll scale by the square root of explained variance to make them proportional to the component's importance
    scale_pc1 = np.sqrt(pca.explained_variance_[pc1-1])
    scale_pc2 = np.sqrt(pca.explained_variance_[pc2-1])

    plt.figure(figsize=(10, 8))

    # Plot the data points (PCA scores), color-coded by 'bandgap'
    scatter = plt.scatter(pca_scores[:, pc1-1], pca_scores[:, pc2-1],
                          c=df_pca['bandgap'], cmap='jet', vmin=0.5, vmax=3.5,
                          alpha=0.75, edgecolors=original_df['color_edge'], s=df_pca['rB'])

    # Add a colorbar for the bandgap values
    plt.colorbar(scatter, label='Bandgap (eV)')

    # Plot the feature loadings as vectors
    for i, feature in enumerate(df_scaled.columns):
        # Adjust the length of the loading vectors for better visibility
        # Here, scaling by a factor relative to the component's variance and the range of scores
        # Using .iloc to avoid FutureWarning
        col = 'red'
        if feature == 'bandgap':
            col = 'blue'
        plt.arrow(0, 0, loadings_pc1.iloc[i] * scale_pc1 * 3, loadings_pc2.iloc[i] * scale_pc2 * 3,
                  head_width=0.05, head_length=0.05, fc=col, ec=col, linewidth=1.5)

        # Highlight 'bandgap' specifically
        if feature == 'bandgap':
            plt.text(loadings_pc1.iloc[i] * scale_pc1 * 3 * 1.2, loadings_pc2.iloc[i] * scale_pc2 * 3 * 1.2,
                    dict_values[feature], color='blue', ha='center', va='center', fontweight='bold')
        else:
            plt.text(loadings_pc1.iloc[i] * scale_pc1 * 3 * 1.2, loadings_pc2.iloc[i] * scale_pc2 * 3 * 1.2,
                    dict_values[feature], color='red', ha='center', va='center')

    plt.xlabel(f'PC {pc1} ({pca.explained_variance_ratio_[pc1-1]*100:.1f}% variance)')
    plt.ylabel(f'PC {pc2} ({pca.explained_variance_ratio_[pc2-1]*100:.1f}% variance)')
    plt.axhline(0, color='k', linewidth=2)
    plt.axvline(0, color='k', linewidth=2)
    plt.xlim([-4.2,4.2])
    plt.ylim([-4.2,4.2])

    name_file = f'PCA_PC{pc1}_PC{pc2}_plot.png'

    plt.savefig(FIGURES_DIR / name_file, dpi=600, bbox_inches='tight')

    plt.show()

def corr_matrix(df: pd.DataFrame, metrics: List[str], dict_labels: Dict[str, str]) -> None:
    """Generate Spearman rank-correlation matrix heatmap for scoring metrics.

    Creates a heatmap of pairwise Spearman correlations (robust to non-normal
    and bounded distributions) with significance annotations.  Prints an
    interpretation summary to the console.

    Args:
        df: DataFrame containing the metric columns to analyze.
        metrics: List of metric column names to include in correlation analysis.
        dict_labels: Dictionary mapping column names to display labels.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import spearmanr

    sns.set_context('talk')

    available_metrics = [m for m in metrics if m in df.columns]
    corr_data = df[available_metrics].dropna()
    n = len(corr_data)

    # --- Spearman rank correlation + two-sided p-values ---
    rho_mat, p_mat = spearmanr(corr_data)

    # spearmanr returns scalars when only 2 variables; ensure 2-D arrays
    if corr_data.shape[1] == 2:
        rho_mat = np.array([[1.0, rho_mat], [rho_mat, 1.0]])
        p_mat   = np.array([[0.0, p_mat],   [p_mat,   0.0]])

    rho_df = pd.DataFrame(rho_mat, index=available_metrics, columns=available_metrics)
    p_df   = pd.DataFrame(p_mat,   index=available_metrics, columns=available_metrics)

    # Build annotation strings: ρ value + significance stars
    def _sig_stars(p: float) -> str:
        if p < 0.001:
            return '***'
        if p < 0.01:
            return '**'
        if p < 0.05:
            return '*'
        return ''

    annot_array = np.empty_like(rho_mat, dtype=object)
    for i in range(len(available_metrics)):
        for j in range(len(available_metrics)):
            stars = _sig_stars(p_df.iloc[i, j])
            annot_array[i, j] = f'{rho_df.iloc[i, j]:.2f}{stars}'

    # Rename for display labels
    display_labels = [dict_labels.get(m, m) for m in available_metrics]

    # --- Heatmap ---
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        rho_df.values,
        annot=annot_array,
        fmt='',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        xticklabels=display_labels,
        yticklabels=display_labels,
        ax=ax,
    )
    ax.set_title(f'Spearman rank correlation (n = {n})', fontsize=14, pad=12)

    plt.savefig(FIGURES_DIR / 'metric_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # --- Console interpretation ---
    print(f"\nSpearman rank correlations (n = {n}):")
    print("-" * 60)
    for i, m1 in enumerate(available_metrics):
        for m2 in available_metrics[i+1:]:
            rho = rho_df.loc[m1, m2]
            p   = p_df.loc[m1, m2]
            strength = "weak" if abs(rho) < 0.3 else "moderate" if abs(rho) < 0.6 else "strong"
            direction = "positive" if rho > 0 else "negative"
            sig = f"p = {p:.4f}" if p >= 0.001 else f"p = {p:.2e}"
            orthogonal = "  → INDEPENDENT" if abs(rho) < 0.3 else ""
            print(f"  {dict_labels.get(m1, m1):>12s} vs {dict_labels.get(m2, m2):<12s}: "
                  f"ρ = {rho:+.3f} ({strength} {direction}, {sig}){orthogonal}")

    print("\n* p < 0.05, ** p < 0.01, *** p < 0.001")
    print("Low |ρ| values indicate metrics capture independent material characteristics.")



def colormap_radii_interactive(df: pd.DataFrame, exp_df: pd.DataFrame, clf_proba: Optional[Any] = None, t_sisso: bool = False, anion: Optional[str] = None) -> Any:
    """Create interactive 2D heatmap of stability predictions vs ionic radii.

    Interactive Plotly version of :func:`colormap_radii`. Generates heatmaps
    showing t_sisso or P(t_sisso) values across the (rA, rB) ionic radii space,
    with experimentally observed compounds as hoverable markers.

    Args:
        df: DataFrame with predicted compositions containing rA, rB, rX columns.
        exp_df: DataFrame with experimental compounds for overlay markers, indexed by material name.
        clf_proba: Pre-trained Platt scaling classifier; if None, loads from file.
        t_sisso: If True, plot raw t_sisso values; if False, plot P(t_sisso).
        anion: If 'S' or 'Se', render a single-panel figure for that anion only.
            If None (default), render both side-by-side.

    Returns:
        plotly.graph_objects.Figure: Interactive figure.
    """
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if clf_proba is None:
        from tf_chpvk_pv.modeling.train import train_platt_scaling

        train_df = pd.read_csv(RESULTS_DIR / "processed_chpvk_train_dataset.csv")
        test_df = pd.read_csv(RESULTS_DIR / "processed_chpvk_test_dataset.csv")

        with open(INTERIM_DATA_DIR / "tolerance_factor_classifiers.pkl", 'rb') as file:
            clfs = pickle.load(file)

        train_df, test_df, clf_proba = train_platt_scaling(train_df, test_df, clfs['t_sisso'])

    rX_range = [df.rX.min(), df.rX.max()]

    def calculate_t_sisso(rA, rB, rX):
        return np.abs(((rA / rX + rB / rX) + (np.abs(rB / rX - np.log(rA / rB)))) - (rA / rX) ** 3)

    rA_vals = np.linspace(110, 180, 500)
    rB_vals = np.linspace(50, 120, 500)
    xv, yv = np.meshgrid(rA_vals, rB_vals)

    t_sisso_S = calculate_t_sisso(xv, yv, rX_range[0])
    t_sisso_Se = calculate_t_sisso(xv, yv, rX_range[1])

    if t_sisso:
        z_S = t_sisso_S
        z_Se = t_sisso_Se
        colorscale = 'RdBu'
        zmin, zmax = 0.5, 2.5
        colorbar_title = 'τ*'
    else:
        z_S = clf_proba.predict_proba(t_sisso_S.reshape(-1, 1))[:, 1].reshape(t_sisso_S.shape)
        z_Se = clf_proba.predict_proba(t_sisso_Se.reshape(-1, 1))[:, 1].reshape(t_sisso_Se.shape)
        colorscale = 'RdBu_r'
        zmin, zmax = 0.0, 1.0
        colorbar_title = 'P(τ*)'

    exp_df = exp_df[exp_df.rX.isin(rX_range)]
    exp_df = exp_df[exp_df.rX != 196]

    def _exp_traces(anion_rX, subplot_col):
        """Build scatter traces for experimental compounds."""
        traces = []
        df_anion = exp_df[exp_df.rX == anion_rX]
        for label_val, marker_sym, group_name in [(1, 'square', 'Perovskite'), (0, 'triangle-up', 'Non-perovskite')]:
            df_group = df_anion[df_anion.exp_label == label_val]
            if df_group.empty:
                continue
            hover = [
                f"<b>{idx}</b><br>rA = {row.rA:.1f} pm<br>rB = {row.rB:.1f} pm"
                for idx, row in df_group.iterrows()
            ]
            traces.append(go.Scatter(
                x=df_group.rA,
                y=df_group.rB,
                mode='markers',
                marker=dict(symbol=marker_sym, size=10, color='lightgray',
                            line=dict(color='black', width=1.5)),
                name=group_name,
                hovertext=hover,
                hoverinfo='text',
                showlegend=(subplot_col == 1),
                legendgroup=group_name,
            ))
        return traces

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['ABS₃ compounds', 'ABSe₃ compounds'],
                        shared_yaxes=True,
                        horizontal_spacing=0.06)

    # Shared kwargs (no showscale — set per-trace below)
    heatmap_kwargs = dict(
        x=rA_vals, y=rB_vals,
        colorscale=colorscale, zmin=zmin, zmax=zmax,
        colorbar=dict(title=colorbar_title, x=1.02),
        hovertemplate='rA = %{x:.1f} pm<br>rB = %{y:.1f} pm<br>' + colorbar_title + ' = %{z:.3f}<extra></extra>',
    )

    if anion is not None:
        # Single-anion figure: full width, no subplots
        rX_val = rX_range[0] if anion == 'S' else rX_range[1]
        z = z_S if anion == 'S' else z_Se
        anion_label = anion
        title_text = f'AB{anion_label}₃ — {colorbar_title}'

        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=z, showscale=True, **heatmap_kwargs))
        for trace in _exp_traces(rX_val, 1):
            fig.add_trace(trace)

        fig.update_xaxes(title_text='r<sub>A</sub> (pm)', range=[110, 180])
        fig.update_yaxes(title_text='r<sub>B</sub> (pm)', range=[50, 120])
        fig.update_layout(
            title=title_text,
            height=540,
            template='plotly_white',
            legend=dict(x=0.01, y=0.01, bgcolor='rgba(255,255,255,0.7)'),
        )
        return fig

    fig.add_trace(go.Heatmap(z=z_S, showscale=False, **heatmap_kwargs), row=1, col=1)
    fig.add_trace(go.Heatmap(z=z_Se, showscale=True, **heatmap_kwargs), row=1, col=2)

    for trace in _exp_traces(rX_range[0], 1):
        fig.add_trace(trace, row=1, col=1)
    for trace in _exp_traces(rX_range[1], 2):
        fig.add_trace(trace, row=1, col=2)

    fig.update_xaxes(title_text='r<sub>A</sub> (pm)', range=[110, 180])
    fig.update_yaxes(title_text='r<sub>B</sub> (pm)', range=[50, 120], row=1, col=1)
    fig.update_layout(
        height=520,
        template='plotly_white',
        legend=dict(x=0.01, y=0.01, bgcolor='rgba(255,255,255,0.7)'),
    )
    return fig


def plot_matrix_interactive(df_out: pd.DataFrame, df_crystal: pd.DataFrame, anion: str = 'S', parameter: str = 'Eg', clf_proba: Optional[Any] = None) -> Any:
    """Create interactive element-element matrix colored by bandgap or P(t_sisso).

    Interactive Plotly version of :func:`plot_matrix`. Generates a matrix scatter
    plot where each point represents a predicted AB composition for the given anion,
    with hover text showing the formula and metric value.
    CrystaLLM-validated compositions are highlighted with a black border.

    Args:
        df_out: DataFrame with predicted compositions containing A, B, X, formula columns.
        df_crystal: DataFrame with CrystaLLM-validated compositions for highlighting.
        anion: Anion element to filter ('S' or 'Se').
        parameter: Coloring parameter ('Eg' for bandgap, 'p_t_sisso' for probability).
        clf_proba: Pre-trained Platt scaling classifier; if None, loads from file.

    Returns:
        plotly.graph_objects.Figure: Interactive element matrix figure.
    """
    import plotly.graph_objects as go

    element_to_number = {
        "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
        "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
        "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
        "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40,
        "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
        "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
        "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
        "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
        "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
        "Pa": 91, "U": 92,
    }

    if parameter == 'p_t_sisso' and clf_proba is None:
        from tf_chpvk_pv.modeling.train import train_platt_scaling

        train_df = pd.read_csv(RESULTS_DIR / "processed_chpvk_train_dataset.csv")
        test_df = pd.read_csv(RESULTS_DIR / "processed_chpvk_test_dataset.csv")

        with open(INTERIM_DATA_DIR / "tolerance_factor_classifiers.pkl", 'rb') as file:
            clfs = pickle.load(file)

        train_df, test_df, clf_proba = train_platt_scaling(train_df, test_df, clfs['t_sisso'])
        df_out = df_out.copy()
        df_out['p_t_sisso'] = clf_proba.predict_proba(df_out['t_sisso'].values.reshape(-1, 1))[:, 1]

    df_out = df_out.copy()
    df_out["norm_formula"] = df_out["formula"].apply(normalize_abx3)
    df_crystal = df_crystal.copy()
    df_crystal["norm_formula"] = df_crystal["formula"].apply(normalize_abx3)
    df_sisso = df_out[df_out['norm_formula'].isin(df_crystal['norm_formula'])]

    df_anion = df_out[df_out['X'] == anion].copy()
    if parameter in ['Eg']:
        df_anion = df_anion[df_anion[parameter] > 0].copy()
    df_anion['Z_A'] = df_anion.A.map(element_to_number)
    df_anion['Z_B'] = df_anion.B.map(element_to_number)
    df_anion.sort_values(by=['Z_A', 'Z_B'], inplace=True, ascending=[True, True])

    df_crystal_anion = df_sisso[df_sisso['X'] == anion].copy()
    if parameter in ['Eg']:
        df_crystal_anion = df_crystal_anion[df_crystal_anion[parameter] > 0].copy()

    if parameter == 'Eg':
        colorscale = 'Jet'
        cmin, cmax = 0.5, 3.5
        colorbar_title = 'Bandgap (eV)'
        hover_label = 'E<sub>g</sub>'
        hover_unit = ' eV'
    else:
        colorscale = 'RdBu_r'
        cmin, cmax = 0.0, 1.0
        colorbar_title = 'P(τ*)'
        hover_label = 'P(τ*)'
        hover_unit = ''

    hover_all = [
        f"<b>{row['formula']}</b><br>{hover_label} = {row[parameter]:.3f}{hover_unit}"
        for _, row in df_anion.iterrows()
    ]
    hover_crystal = [
        f"<b>{row['formula']}</b><br>{hover_label} = {row[parameter]:.3f}{hover_unit}<br><i>CrystaLLM: perovskite-type</i>"
        for _, row in df_crystal_anion.iterrows()
    ]

    # Build sorted axis labels
    b_elements = df_anion['B'].unique().tolist()
    a_elements = df_anion['A'].unique().tolist()
    b_order = sorted(b_elements, key=lambda e: element_to_number.get(e, 999))
    a_order = sorted(a_elements, key=lambda e: element_to_number.get(e, 999))

    # Build 2-D z/hover grids for the heatmap
    import numpy as np
    z_grid = np.full((len(a_order), len(b_order)), float('nan'))
    hover_grid = [["" for _ in b_order] for _ in a_order]
    crystal_set = set(df_crystal_anion['formula'].tolist())

    b_idx = {e: i for i, e in enumerate(b_order)}
    a_idx = {e: i for i, e in enumerate(a_order)}

    for _, row in df_anion.iterrows():
        ai, bi = a_idx[row['A']], b_idx[row['B']]
        z_grid[ai][bi] = row[parameter]
        hover_grid[ai][bi] = f"<b>{row['formula']}</b><br>{hover_label} = {row[parameter]:.3f}{hover_unit}"

    # Crystal border overlay — shapes drawn after layout; keep refs for iteration
    cx = [row['B'] for _, row in df_crystal_anion.iterrows()]
    cy = [row['A'] for _, row in df_crystal_anion.iterrows()]

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=b_order,
        y=a_order,
        z=z_grid,
        colorscale=colorscale,
        zmin=cmin, zmax=cmax,
        colorbar=dict(title=colorbar_title),
        hovertext=hover_grid,
        hoverinfo='text',
        xgap=1, ygap=1,
    ))

    # Legend-only dummy trace for CrystaLLM compounds (borders are drawn as shapes)
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(symbol='square', size=10, color='rgba(0,0,0,0)',
                    line=dict(color='black', width=2)),
        name='CrystaLLM: perovskite-type',
        showlegend=True,
    ))

    # Invisible trace just for the legend entry for heatmap cells
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(symbol='square', size=10, color='navy'),
        name='All τ*-stable',
        showlegend=True,
    ))

    anion_label = 'S' if anion == 'S' else 'Se'
    title_text = f'AB{anion_label}₃ — {colorbar_title}'

    # Black border rectangles using axis coordinates — each heatmap cell spans ±0.5
    shapes = []
    for _, row in df_crystal_anion.iterrows():
        if row['B'] in b_idx and row['A'] in a_idx:
            bi = b_idx[row['B']]
            ai = a_idx[row['A']]
            shapes.append(dict(
                type='rect',
                xref='x', yref='y',
                x0=bi - 0.5, x1=bi + 0.5,
                y0=ai - 0.5, y1=ai + 0.5,
                line=dict(color='black', width=2),
                fillcolor='rgba(0,0,0,0)',
            ))

    fig.update_layout(
        title=title_text,
        xaxis=dict(title='Cation B', tickangle=90, type='category',
                   categoryorder='array', categoryarray=b_order),
        yaxis=dict(title='Cation A', type='category',
                   categoryorder='array', categoryarray=a_order),
        template='plotly_white',
        height=700,
        shapes=shapes,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)'),
    )
    return fig


def pareto_front_interactive(df: pd.DataFrame, variable: str, Eg_ref: float = 1.34,
                              plot_PCE: bool = False,
                              sj_limit_path: Path = RAW_DATA_DIR / "SJ_limit.csv",
                              dj_limit_path: Path = RAW_DATA_DIR / "DJ_limit.csv") -> Any:
    """Create interactive Pareto front scatter plot for bandgap deviation vs sustainability metric.

    Interactive Plotly version of :func:`pareto_front_plot`. Points are colored
    by their Pareto status (gray=dominated, blue=near-optimal, red=Pareto-optimal)
    with hover text showing formula, bandgap, and metric value.

    Args:
        df: DataFrame with 'bandgap', 'formula', 'A', 'B', and sustainability metric columns.
        variable: Sustainability metric column name ('HHI', 'SR', or '1-CL score').
        Eg_ref: Reference bandgap in eV (1.34 for single-junction, 1.71 for tandem).
        plot_PCE: If True, overlay theoretical PCE limit colormap background.
        sj_limit_path: Path to single-junction Shockley-Queisser limit CSV.
        dj_limit_path: Path to dual-junction limit CSV for tandem cells.

    Returns:
        plotly.graph_objects.Figure: Interactive Pareto front figure.
    """
    import numpy as np
    import plotly.graph_objects as go
    import pandas as pd

    df = df[df['B'] != 'U']
    df = df[df['A'] != 'U'].reset_index(drop=True).copy()
    cols = [variable, 'Eg_dev']
    df["Eg_dev"] = abs(Eg_ref - df["bandgap"])

    is_pareto = np.ones(len(df), dtype=bool)
    for i, point in df[cols].iterrows():
        if is_pareto[i]:
            is_pareto[is_pareto] = ~(
                (df.loc[is_pareto, cols] >= point).all(axis=1) &
                (df.loc[is_pareto, cols] > point).any(axis=1)
            )
            is_pareto[i] = True

    df['color_group'] = 'Dominated'
    df.loc[df['Eg_dev'] / Eg_ref <= 0.10, 'color_group'] = 'Near-optimal'
    var_min, var_max = df[variable].min(), df[variable].max()
    if var_max > var_min:
        df.loc[(df[variable] - var_min) / (var_max - var_min) <= 0.10, 'color_group'] = 'Near-optimal'
    df.loc[is_pareto, 'color_group'] = 'Pareto-optimal'

    variable_title = {
        'HHI': 'Herfindahl-Hirschman Index (HHI)',
        'SR': 'Supply Risk (SR)',
        '1-CL score': '1 − Crystal-likeness Score (CLS)',
    }

    def _hover(row):
        bg_sigma = f" ± {row['bandgap_sigma']:.3f}" if 'bandgap_sigma' in row.index else ''
        return (f"<b>{row['formula']}</b><br>"
                f"E<sub>g</sub> = {row['bandgap']:.3f}{bg_sigma} eV<br>"
                f"|{Eg_ref:.2f} − E<sub>g</sub>| = {row['Eg_dev']:.3f} eV<br>"
                f"{variable} = {row[variable]:.3f}")

    groups = [
        ('Dominated', 'circle', 'rgba(80,80,80,0.6)', 'gray'),
        ('Near-optimal', 'square', 'rgba(30,100,200,0.8)', 'blue'),
        ('Pareto-optimal', 'triangle-up', 'rgba(200,30,30,0.9)', 'red'),
    ]

    fig = go.Figure()

    if plot_PCE:
        try:
            limit_df = pd.read_csv(sj_limit_path if Eg_ref == 1.34 else dj_limit_path)
            pce_vals = limit_df['PCE (%)'].values
            xlims = [0, df["Eg_dev"].max() * 1.1]
            ylims = [df[variable].min() * 0.9, df[variable].max() * 1.1]
            if Eg_ref == 1.34:
                colorscale_pce = [[0, 'rgba(255,255,255,0)'], [1, 'rgba(0,150,80,0.35)']]
                cmin_pce, cmax_pce = 0, 35
            else:
                colorscale_pce = [[0, 'rgba(255,255,255,0)'], [1, 'rgba(0,80,180,0.35)']]
                cmin_pce, cmax_pce = 29, 45
            fig.add_trace(go.Heatmap(
                z=[pce_vals],
                x=list(xlims),
                y=list(ylims),
                colorscale=colorscale_pce,
                zmin=cmin_pce, zmax=cmax_pce,
                colorbar=dict(title='PCE limit (%)', x=1.12, len=0.5, y=0.8),
                hoverinfo='skip',
                showscale=True,
            ))
        except FileNotFoundError:
            pass

    for group_name, marker_sym, color, _ in groups:
        df_g = df[df['color_group'] == group_name]
        if df_g.empty:
            continue
        fig.add_trace(go.Scatter(
            x=df_g['Eg_dev'],
            y=df_g[variable],
            mode='markers',
            marker=dict(symbol=marker_sym, size=10, color=color,
                        line=dict(color='black', width=0.8)),
            name=group_name,
            hovertext=[_hover(row) for _, row in df_g.iterrows()],
            hoverinfo='text',
        ))

    arch = 'Single junction' if Eg_ref == 1.34 else 'Tandem top cell'
    fig.update_layout(
        title=f'{variable_title.get(variable, variable)} — {arch} (E<sub>g</sub><sup>opt</sup> = {Eg_ref} eV)',
        xaxis=dict(title=f'|{Eg_ref:.2f} − E<sub>g</sub>| (eV)', rangemode='tozero'),
        yaxis=dict(title=variable_title.get(variable, variable), rangemode='tozero'),
        template='plotly_white',
        height=520,
        legend=dict(x=0.65, y=0.98, bgcolor='rgba(255,255,255,0.7)'),
    )
    return fig


def corr_matrix_interactive(df: pd.DataFrame, metrics: List[str], dict_labels: Dict[str, str]) -> Any:
    """Create interactive Spearman rank correlation matrix heatmap.

    Interactive Plotly version of :func:`corr_matrix`. Generates an annotated
    heatmap of pairwise Spearman correlations between screening metrics with
    hover text showing exact ρ values.

    Note:
        Uses Spearman rank correlation (consistent with the paper), whereas the
        static :func:`corr_matrix` uses Pearson correlation.

    Args:
        df: DataFrame containing the metric columns to analyze.
        metrics: List of metric column names to include in correlation analysis.
        dict_labels: Dictionary mapping column names to display labels.

    Returns:
        plotly.graph_objects.Figure: Interactive correlation heatmap.
    """
    import plotly.graph_objects as go
    import numpy as np

    available_metrics = [m for m in metrics if m in df.columns]
    corr_data = df[available_metrics].dropna()
    corr = corr_data.corr(method='spearman')

    labels = [dict_labels.get(m, m) for m in available_metrics]
    z = corr.values
    n = len(labels)

    # Build annotation text
    text = [[f'ρ = {z[i][j]:.3f}' for j in range(n)] for i in range(n)]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale='RdBu_r',
        zmin=-1, zmax=1,
        text=text,
        texttemplate='%{text}',
        hovertemplate='%{y} vs %{x}<br>%{text}<extra></extra>',
        colorbar=dict(title='ρ (Spearman)'),
    ))

    fig.update_layout(
        title='Spearman rank correlation — screening metrics',
        xaxis=dict(tickangle=30),
        template='plotly_white',
        height=480,
        width=540,
    )
    return fig


# ---------------------------------------------------------------------------
# Crystal structure helpers
# ---------------------------------------------------------------------------

_CHALCOGENS = {'S', 'Se', 'Te'}

# CPK-inspired colors for elements common in this dataset
_ELEMENT_COLORS: Dict[str, str] = {
    # Anions
    'S':  '#f5c542', 'Se': '#e89b3a',
    # Common A-site
    'Ba': '#1abc9c', 'Sr': '#16a085', 'Ca': '#148f77',
    'Eu': '#2ecc71', 'Yb': '#27ae60', 'Sm': '#1e8449',
    'La': '#52be80', 'Ce': '#45b39d', 'Pr': '#a9cce3',
    'Gd': '#76d7c4', 'Dy': '#5dade2', 'Tb': '#85c1e9',
    # Common B-site
    'Zr': '#4c7be1', 'Hf': '#2e4db9', 'Ti': '#5b8dd9',
    'Sc': '#7fb3f5', 'In': '#6c9ee0', 'Sn': '#8daee8',
    'U':  '#9b59b6', 'Bi': '#8e44ad',
    'Zn': '#aab7b8', 'Cu': '#ca6f1e', 'Cd': '#b7950b',
    'Lu': '#717d7e', 'Tm': '#839192', 'Al': '#abebc6',
}
_DEFAULT_COLOR = '#cccccc'


def _site_roles(structure: Any) -> tuple:
    """Return (role_by_index, a_element, b_element, x_element) for a perovskite structure.

    X-sites are chalcogens (S/Se/Te). B (octahedral) and A (cuboctahedral) are
    distinguished by their average nearest-neighbor distance to X: the cation with
    the shorter average B–X distance is assigned to the B-site.

    Args:
        structure: :class:`pymatgen.core.Structure` object.

    Returns:
        tuple: (role_map, a_el, b_el, x_el) where role_map is a dict
            ``{site_index: 'A'|'B'|'X'}``.
    """
    from scipy.spatial.distance import cdist

    x_idx = [i for i, s in enumerate(structure) if s.species_string in _CHALCOGENS]
    cat_idx = [i for i, s in enumerate(structure) if s.species_string not in _CHALCOGENS]

    x_coords = np.array([structure[i].coords for i in x_idx])
    cation_els = list({structure[i].species_string for i in cat_idx})

    if len(cation_els) == 1:
        role_map = {i: 'X' for i in x_idx}
        role_map.update({i: 'B' for i in cat_idx})
        return role_map, cation_els[0], cation_els[0], structure[x_idx[0]].species_string

    # Shorter mean nearest-X distance → B-site
    avg_dist: Dict[str, float] = {}
    for el in cation_els:
        el_coords = np.array([structure[i].coords for i in cat_idx
                               if structure[i].species_string == el])
        d = cdist(el_coords, x_coords)
        avg_dist[el] = float(d.min(axis=1).mean())

    b_el = min(avg_dist, key=avg_dist.get)
    a_el = next(el for el in cation_els if el != b_el)
    x_el = structure[x_idx[0]].species_string

    role_map: Dict[int, str] = {}
    for i in x_idx:
        role_map[i] = 'X'
    for i in cat_idx:
        role_map[i] = 'B' if structure[i].species_string == b_el else 'A'

    return role_map, a_el, b_el, x_el


def plot_crystal_structure_interactive(
    cif_path: Any,
    supercell: tuple = (1, 1, 1),
    alpha: float = 0.35,
    bx_cutoff: float = 4.0,
) -> Any:
    """Create an interactive 3D polyhedral view of a perovskite crystal structure.

    Reads a CIF file with pymatgen, identifies A/B/X sites, draws BX₆ octahedra
    as semi-transparent Plotly ``Mesh3d`` traces, and overlays atom spheres with
    hover labels.  The unit-cell box is drawn as a black wire frame.

    Args:
        cif_path: Path to the CIF file.
        supercell: Supercell dimensions as ``(na, nb, nc)``. Default ``(1, 1, 1)``
            (the as-read unit cell, which contains Z = 4 formula units for Pnma).
        alpha: Opacity of the polyhedral faces (0 = transparent, 1 = opaque).
        bx_cutoff: Neighbor search radius (Å) used to find X-site atoms around
            each B-site atom for constructing octahedral polyhedra.

    Returns:
        plotly.graph_objects.Figure: Interactive crystal structure figure.
    """
    import plotly.graph_objects as go
    from pymatgen.core import Structure
    from scipy.spatial import ConvexHull

    structure = Structure.from_file(str(cif_path))
    if list(supercell) != [1, 1, 1]:
        structure.make_supercell(list(supercell))

    role_map, a_el, b_el, x_el = _site_roles(structure)

    # -------------------------------------------------------------------------
    # Polyhedral traces — one Mesh3d per B atom (BX₆ octahedron)
    # -------------------------------------------------------------------------
    b_indices = [i for i, r in role_map.items() if r == 'B']

    # Use pymatgen periodic neighbor search so atoms at cell boundaries are correct
    all_neighbors = structure.get_all_neighbors(r=bx_cutoff)

    poly_vx: List[float] = []
    poly_vy: List[float] = []
    poly_vz: List[float] = []
    poly_fi: List[int] = []
    poly_fj: List[int] = []
    poly_fk: List[int] = []
    vertex_offset = 0

    for bi in b_indices:
        x_neighbors = [
            n for n in all_neighbors[bi]
            if n.species_string in _CHALCOGENS
        ]
        # Sort by distance, keep 6 nearest
        x_neighbors.sort(key=lambda n: n.nn_distance)
        verts = np.array([n.coords for n in x_neighbors[:6]])

        if len(verts) < 4:
            continue
        try:
            hull = ConvexHull(verts)
            poly_vx.extend(verts[:, 0])
            poly_vy.extend(verts[:, 1])
            poly_vz.extend(verts[:, 2])
            for simplex in hull.simplices:
                poly_fi.append(vertex_offset + int(simplex[0]))
                poly_fj.append(vertex_offset + int(simplex[1]))
                poly_fk.append(vertex_offset + int(simplex[2]))
            vertex_offset += len(verts)
        except Exception:
            pass

    traces: List[Any] = []

    b_color = _ELEMENT_COLORS.get(b_el, _DEFAULT_COLOR)
    if poly_vx:
        traces.append(go.Mesh3d(
            x=poly_vx, y=poly_vy, z=poly_vz,
            i=poly_fi, j=poly_fj, k=poly_fk,
            color=b_color,
            opacity=alpha,
            name=f'{b_el}X₆ octahedra',
            hoverinfo='skip',
            showlegend=True,
            flatshading=False,
            lighting=dict(ambient=0.7, diffuse=0.8, specular=0.3, roughness=0.5),
            lightposition=dict(x=100, y=200, z=100),
        ))

    # -------------------------------------------------------------------------
    # Atom sphere traces — one Scatter3d per element
    # -------------------------------------------------------------------------
    marker_sizes = {a_el: 12, b_el: 8, x_el: 6}
    for el, role in [(a_el, 'A'), (b_el, 'B'), (x_el, 'X')]:
        idx = [i for i, r in role_map.items() if r == role]
        if not idx:
            continue
        coords = np.array([structure[i].coords for i in idx])
        color = _ELEMENT_COLORS.get(el, _DEFAULT_COLOR)
        traces.append(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode='markers',
            marker=dict(
                size=marker_sizes.get(el, 8),
                color=color,
                line=dict(color='black', width=0.5),
            ),
            name=f'{el} ({role}-site)',
            hovertemplate=(
                f'<b>{el} ({role}-site)</b><br>'
                'x = %{x:.2f} Å<br>y = %{y:.2f} Å<br>z = %{z:.2f} Å'
                '<extra></extra>'
            ),
        ))

    # -------------------------------------------------------------------------
    # Unit-cell wire frame (12 edges of the parallelepiped)
    # -------------------------------------------------------------------------
    latt = structure.lattice.matrix  # rows are a, b, c vectors
    o = np.zeros(3)
    av, bv, cv = latt[0], latt[1], latt[2]
    corners = [
        o, av, bv, cv,
        av + bv, av + cv, bv + cv, av + bv + cv,
    ]
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7),
    ]
    ex: List[Optional[float]] = []
    ey: List[Optional[float]] = []
    ez: List[Optional[float]] = []
    for e0, e1 in edges:
        ex += [float(corners[e0][0]), float(corners[e1][0]), None]
        ey += [float(corners[e0][1]), float(corners[e1][1]), None]
        ez += [float(corners[e0][2]), float(corners[e1][2]), None]

    traces.append(go.Scatter3d(
        x=ex, y=ey, z=ez,
        mode='lines',
        line=dict(color='black', width=2),
        name='Unit cell',
        hoverinfo='skip',
    ))

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------
    formula = structure.composition.reduced_formula
    sc_label = '×'.join(str(s) for s in supercell)
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f'{formula} — polyhedral view ({sc_label} supercell)',
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False,
                       showgrid=False, zeroline=False, title=''),
            yaxis=dict(showbackground=False, showticklabels=False,
                       showgrid=False, zeroline=False, title=''),
            zaxis=dict(showbackground=False, showticklabels=False,
                       showgrid=False, zeroline=False, title=''),
            aspectmode='data',
            bgcolor='white',
        ),
        template='plotly_white',
        height=600,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)'),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


if __name__ == "__main__":
    app()
