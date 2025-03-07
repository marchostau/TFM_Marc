import os
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors

from .utils import concatenate_datasets
from ..logging_information.logging_config import get_logger

logger = get_logger(__name__)


class OutliersRemovalMode(str, Enum):
    IQR = 'iqr'
    DBSCAN = 'dbscan'


def compute_epsilon(dataframe: pd.DataFrame, n_neighbours: int, cols: list):
    nbrs = NearestNeighbors(
        n_neighbors=n_neighbours
    ).fit(dataframe[cols])

    logger.info(f"Dataframe in compute epsilon:\n{dataframe}")
    logger.info(f"N_neighbours: {n_neighbours}")
    logger.info(f"Cols: {cols}")

    neigh_dist, neigh_ind = nbrs.kneighbors(
        dataframe[cols]
    )

    sorted_neigh_dist = np.sort(neigh_dist, axis=0)
    k_dist = sorted_neigh_dist[:, -1]

    knee_point = KneeLocator(
        x=range(1, len(neigh_dist)+1), y=k_dist,
        S=1.0, curve="concave", direction="increasing",
        online=True
    )
    return knee_point.knee_y


def remove_outliers_dbscan(
        dataframe: pd.DataFrame, cols: list,
        eps: float = None, min_samples: int = None
):
    min_samples = 2*len(cols) if min_samples is None else min_samples
    if len(dataframe) < min_samples:
        logger.warning(
            f"Can't apply remove outliers to this segment because "
            f"it has {len(dataframe)} rows and min_samples is {min_samples}"
        )
        return None
    eps = compute_epsilon(dataframe, min_samples, cols) if eps is None else eps
    if eps is None:
        logger.warning(
            f"Epsilon computed is None with eps={eps}, rows={len(dataframe)}" 
            f", min_samples={min_samples} on columns: {cols}"
        )
        return None

    logger.info(
        f"Applying DBSCAN with eps={eps}, " 
        f"min_samples={min_samples} on columns: {cols}"
    )

    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(dataframe[cols])
    dataframe['cluster'] = dbscan.labels_

    original_len = len(dataframe)
    dataframe = dataframe[
            dataframe['cluster'] != -1
        ].drop(columns=['cluster'])
    logger.info(f"DBSCAN removed {original_len - len(dataframe)} outliers")

    return dataframe


def compute_iqr_bounds(dataframe: pd.DataFrame, cols: list):
    bounds = {}
    for col in cols:
        q1, q3 = dataframe[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return bounds


def remove_outliers_iqr(
        dataframe: pd.DataFrame, cols: list, bounds: dict = None
):
    if not bounds:
        bounds = compute_iqr_bounds(dataframe, cols)

    logger.info(
        f"Applying IQR-based outlier removal on columns: "
        f"{cols} with bounds: {bounds}"
    )
    original_len = len(dataframe)
    for col in cols:
        lower_bound, upper_bound = bounds[col]
        dataframe = dataframe[
            (dataframe[col] >= lower_bound) &
            (dataframe[col] <= upper_bound)
        ]
    logger.info(f"IQR removed {original_len - len(dataframe)} outliers")
    return dataframe


def remove_outliers_dataset(
        dataframe: pd.DataFrame, mode: OutliersRemovalMode,
        cols: list, **kwargs
):
    mode_parameters = kwargs.get("mode_parameters", {})
    logger.info(f"Removing outliers using mode: {mode.value}")

    match mode:
        case OutliersRemovalMode.IQR:
            return remove_outliers_iqr(
                dataframe, cols, bounds=mode_parameters.get("bounds", None)
            )

        case OutliersRemovalMode.DBSCAN:
            return remove_outliers_dbscan(
                dataframe, cols,
                eps=mode_parameters.get("eps", None),
                min_samples=mode_parameters.get("min_samples", None)
            )

        case _:
            logger.error(f"Invalid mode specified: {mode}")
            raise ValueError(f"Invalid mode: {mode}")


def compute_outlier_params(
        dataframe: pd.DataFrame, mode: OutliersRemovalMode, cols: list
):
    mode_parameters = {}

    match mode:
        case OutliersRemovalMode.DBSCAN:
            min_samples = 2 * len(cols)
            eps = compute_epsilon(dataframe, min_samples)
            mode_parameters = {"min_samples": min_samples, "eps": eps}

        case OutliersRemovalMode.IQR:
            return {"bounds": compute_iqr_bounds(dataframe, cols)}

        case _:
            raise ValueError(f"Invalid mode: {mode}")

    return mode_parameters


def remove_outliers_dir_dataset(
        dir_source: str, dir_output: str,
        mode: OutliersRemovalMode, cols: list,
        daily_based: bool
):
    if not os.path.isdir(dir_source):
        logger.error(f"Invalid directory: {dir_source}")
        raise ValueError(f'{dir_source} is not a directory')

    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
        logger.info(f"Created output directory: {dir_output}")

    mode_parameters = {}
    if not daily_based:
        logger.info("Data processing based on global strategy.")
        logger.info("Concatenating datasets for global outlier detection.")
        complete_df = concatenate_datasets(dir_source)
        if complete_df.empty:
            logger.warning(
                "No valid data found in the source directory. "
                "Skipping normalization."           
            )
            raise ValueError(
                "No valid data found in the source directory. "
                "Skipping normalization."
            )
        mode_parameters = compute_outlier_params(complete_df, mode, cols)
    else:
        logger.info("Data processing based on daily strategy.")

    files = sorted(os.listdir(dir_source))
    for filename in files:
        file_path = os.path.join(dir_source, filename)
        if os.path.isfile(file_path):
            logger.info(f"Processing file for outlier removal: {filename}")
            try:
                dataframe = pd.read_csv(file_path, parse_dates=["timestamp"])
            except ValueError:
                logger.error(f"Error reading CSV file: {filename}")
                continue
            cleaned_df = remove_outliers_dataset(
                dataframe, mode, cols, mode_parameters=mode_parameters
            )

            output_filepath = os.path.join(dir_output, filename)
            cleaned_df.to_csv(output_filepath, index=False)
            logger.info(f"Processed file saved: {output_filepath}")


def remove_wrong_timestamps(dataframe: pd.DataFrame):
    logger.info("Removing wrong timestamps")
    original_len = len(dataframe)
    mode_timestamp = dataframe['timestamp'].dt.day.mode()[0]
    dataframe = dataframe[dataframe['timestamp'].dt.day == mode_timestamp]
    logger.info(f"Removed {original_len - len(dataframe)} duplicated rows")
    return dataframe


def remove_repeated_timestamps(dataframe: pd.DataFrame):
    logger.info("Removing duplicate timestamps")
    original_len = len(dataframe)
    dataframe = dataframe.drop_duplicates(
        subset=['timestamp'], keep='first'
    )
    logger.info(f"Removed {original_len - len(dataframe)} duplicated rows")
    return dataframe


def remove_repeated_timestamps_dir(dir_source: str, dir_output: str):
    if not os.path.isdir(dir_source):
        logger.error(f"Invalid directory: {dir_source}")
        raise ValueError(f'{dir_source} is not a directory')

    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
        logger.info(f"Created output directory: {dir_output}")

    files = sorted(os.listdir(dir_source))
    for filename in files:
        if os.path.isfile(filename):
            logger.info(
                f"Processing file for repeated timestamps removal: {filename}"
            )
            file_path = os.path.join(dir_source, filename)
            try:
                dataframe = pd.read_csv(file_path, parse_dates=["timestamp"])
            except ValueError:
                logger.error(f"Error reading CSV file: {filename}")
                continue
            cleaned_df = remove_repeated_timestamps(dataframe)
            output_filepath = os.path.join(dir_output, filename)
            cleaned_df.to_csv(output_filepath, index=False)
            logger.info(
                "The repeated timestamps have been removed "
                f"and the file is saved in: {output_filepath}"
            )


def split_continuous_segments(
        dataframe: pd.DataFrame, gap_threshold: pd.Timedelta
):
    logger.info(
        f"Splitting not continous timestamps with a gap {gap_threshold}"
    )
    time_diffs = dataframe["timestamp"].diff()
    group_ids = (time_diffs > gap_threshold).cumsum()
    segments = [group for _, group in dataframe.groupby(group_ids)] 
    logger.info(f"Dataframe splitted in {len(segments)} segments")
    return segments


def average_sliding_window(dataframe: pd.DataFrame, window: pd.Timedelta):
    logger.info(f"Applying sliding window average with a window of {window}")
    original_len = len(dataframe)
    dataframe = dataframe.set_index("timestamp")

    cols_to_average = dataframe.columns.difference(['wind_flag', 'file_name'])
    averaged_df = dataframe[
        cols_to_average
    ].resample(window).mean()
    non_averaged_df = dataframe[
        ['wind_flag', 'file_name']
    ].resample(window).first()

    result_df = pd.concat(
        [averaged_df, non_averaged_df], axis=1
    ).reset_index()

    result_df = result_df.dropna()

    logger.info(
        f"Applied sliding window average with a window of {window}. We've " 
        f"passed from {original_len} rows to {len(result_df)}"
    )
    return result_df
