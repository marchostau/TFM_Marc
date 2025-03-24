import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import seaborn as sns


def compute_histograms(
        dataframe: pd.DataFrame, num_bins: int, 
        columns: list, dir_output: str = None, 
        file_output: str = None, save: bool = False
):
    data_columns = {}
    for col in columns:
        data_columns[col] = dataframe[col]

    num_cols = len(columns)
    num_rows = math.ceil(num_cols ** 0.5)
    num_cols = math.ceil(num_cols / num_rows)

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols)

    histogram_results = {}
    for (ax, (title, data)) in zip(axes.flat, data_columns.items()):
        counts, bin_edges = np.histogram(data, bins=num_bins)
        ax.hist(data, bins=num_bins, color='skyblue', edgecolor='black')
        ax.set_title(title)
        ax.set_xlabel('Values')
        ax.set_ylabel('Frequency')

        histogram_results[title] = {
            "Bin Ranges": bin_edges.tolist(),
            "Counts": counts.tolist()
        }

    plt.tight_layout()
    if save:
        path_output = os.path.join(dir_output, file_output)
        plt.savefig(f"{path_output}.png")
    else:
        plt.show()
    plt.close()

    for key, values in histogram_results.items():
        print(f"Histogram for {key}:")
        print(f"Bin Ranges: {values['Bin Ranges']}")
        print(f"Counts: {values['Counts']}\n")


def compute_granger_causality_tests(
        df: pd.DataFrame, cols: list = ['u_component', 'v_component']
):
    print(f"Granger test for {cols[0]}-{cols[1]}")
    grangercausalitytests(df[cols], maxlag=12)
    print(f"Granger test for {cols[1]}-{cols[0]}")
    grangercausalitytests(df[[cols[1], cols[0]]], maxlag=12)


def analyze_data(df: pd.DataFrame, variable: str):
    df.set_index('timestamp', inplace=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Time Series Plot
    axes[0].plot(
        df.index, df[variable], marker='o', linestyle='-', color='blue'
    )
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel(variable)
    axes[0].set_title(f'Time Series Plot of {variable}')
    axes[0].grid()

    # Histogram with KDE Plot
    sns.histplot(
        df[variable], kde=True, bins=10,
        color='blue', edgecolor='black', ax=axes[1]
    )
    axes[1].set_xlabel(variable)
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Histogram of {variable} with KDE')
    axes[1].grid()

    # Adjust layout
    plt.tight_layout()
    plt.show()

    # Perform Augmented Dickey-Fuller (ADF) Test
    adf_result = adfuller(df[variable])
    print(f"ADF Statistic: {adf_result[0]}")
    print(f"p-value: {adf_result[1]}")
    if adf_result[1] <= 0.05:
        print(
            f"The time series '{variable}' is "
            "stationary (reject the null hypothesis).")
    else:
        print(
            f"The time series '{variable}' is NOT "
            "stationary (fail to reject the null hypothesis)."
        )

    compute_granger_causality_tests(df)
