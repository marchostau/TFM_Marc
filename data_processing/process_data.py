import os

import pandas as pd

from .data_cleaning import (
    remove_wrong_timestamps,
    remove_repeated_timestamps,
    remove_outliers_dataset,
    compute_outlier_params,
    split_continuous_segments,
    average_sliding_window,
)
from .normalization import normalize_dataset
from .utils import concatenate_datasets
from .config_schema import load_config
from ..logging_information.logging_config import get_logger

logger = get_logger(__name__)


def process_datasets(config_path: str):
    p_config = load_config(config_path)

    if not os.path.isdir(p_config.dir_source):
        logger.error(f"Invalid directory: {p_config.dir_source}")
        raise ValueError(f"'{p_config.dir_source}' is not a directory")

    if not os.path.exists(p_config.dir_output):
        os.makedirs(p_config.dir_output)
        logger.info(f"Created output directory: {p_config.dir_output}")

    outliers_mode_parameters = {}
    if p_config.remove_outliers and not p_config.outlier_daily_based:
        logger.info("Computing global outlier removal parameters.")
        complete_df = concatenate_datasets(p_config.dir_source)
        if complete_df.empty:
            raise ValueError("No valid data found in the source directory.")
        outliers_mode_parameters = compute_outlier_params(
            complete_df, p_config.outlier_mode, p_config.outlier_cols
        )

    norm_mode_parameters = {}
    if p_config.normalize and not p_config.norm_daily_based:
        logger.info("Computing global normalization parameters.")
        complete_df = concatenate_datasets(p_config.dir_source)
        if complete_df.empty:
            raise ValueError("No valid data found in the source directory.")
        norm_mode_parameters["mean"] = complete_df[p_config.norm_cols].mean()
        norm_mode_parameters["std"] = complete_df[p_config.norm_cols].std()

    files = sorted(os.listdir(p_config.dir_source))
    for filename in files:
        file_path = os.path.join(p_config.dir_source, filename)
        if os.path.isfile(file_path):
            logger.info(f"Processing file: {filename}")
            try:
                dataframe = pd.read_csv(file_path, parse_dates=["timestamp"])
            except ValueError:
                logger.warning(
                    f"Warning: CSV '{filename}' could not be read, skipping..."
                )
                continue

            if p_config.remove_duplicates:
                dataframe = remove_repeated_timestamps(dataframe)

            if p_config.remove_wrong_dates:
                dataframe = remove_wrong_timestamps(dataframe)

            if p_config.split_continuous_segments:
                segments = split_continuous_segments(
                    dataframe, pd.Timedelta(p_config.gap_threshold)
                )
            else:
                segments = [dataframe]

            for i, segment in enumerate(segments):
                if p_config.normalize:
                    segment = normalize_dataset(
                        segment,
                        p_config.norm_mode,
                        p_config.norm_cols,
                        mode_parameters=norm_mode_parameters
                    )

                if p_config.remove_outliers:
                    segment = remove_outliers_dataset(
                        segment,
                        p_config.outlier_mode,
                        p_config.outlier_cols,
                        mode_parameters=outliers_mode_parameters
                    )
                    if segment is None:
                        continue

                if p_config.sliding_window:
                    segment = average_sliding_window(
                        segment, pd.Timedelta(p_config.window_size)
                    )

                segment_suffix = (
                    f"_segment_{i+1}" if p_config.split_continuous_segments
                    else ""
                )
                output_filename = (
                    f"{filename.replace('.csv', '')}_processed"
                    f"{segment_suffix}.csv"
                )
                output_filepath = os.path.join(
                    p_config.dir_output, output_filename
                )

                segment.to_csv(output_filepath, index=False)
                logger.info(f"Processed file saved: {output_filepath}")
