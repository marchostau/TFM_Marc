import os
import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
