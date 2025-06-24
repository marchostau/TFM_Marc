import os

import pandas as pd
import numpy as np

from .data_cleaning import (
    remove_wrong_timestamps,
    remove_repeated_timestamps,
    remove_outliers_dataset,
    remove_points_outside_polygon,
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
        global_mean = complete_df[p_config.norm_cols].mean()
        global_std = complete_df[p_config.norm_cols].std()
        norm_mode_parameters["mean"] = global_mean
        norm_mode_parameters["std"] = global_std
        logger.info(f"Global mean: {global_mean} | std: {global_std}")

    dbscan_stats = {
        'files_processed': 0,
        'total_rows_before': 0,
        'total_rows_removed': 0,
        'per_file_removals': []
    }

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

            if p_config.remove_points_outside_polygon:
                dataframe = remove_points_outside_polygon(dataframe)

            if p_config.split_continuous_segments:
                segments = split_continuous_segments(
                    dataframe, pd.Timedelta(p_config.gap_threshold)
                )
            else:
                segments = [dataframe]

            for i, segment in enumerate(segments):
                original_len = len(segment)

                segment_suffix = (
                    f"_segment_{i+1}" if p_config.split_continuous_segments
                    else ""
                )

                output_filename = (
                    f"{filename.replace('.csv', '')}_processed"
                    f"{segment_suffix}.csv"
                )
                
                if p_config.normalize:
                    segment = normalize_dataset(
                        segment,
                        p_config.norm_mode,
                        p_config.norm_cols,
                        mode_parameters=norm_mode_parameters
                    )
                
                if p_config.remove_outliers:
                    segment, removed_rows = remove_outliers_dataset(
                        segment,
                        p_config.outlier_mode,
                        p_config.outlier_cols,
                        mode_parameters=outliers_mode_parameters
                    )
                    if segment is None:
                        continue

                    removal_percentage = (removed_rows / original_len) * 100
                    dbscan_stats['files_processed'] += 1
                    dbscan_stats['total_rows_before'] += original_len
                    dbscan_stats['total_rows_removed'] += removed_rows

                    file_stats = {
                        'filename': filename,
                        'segment': i+1 if p_config.split_continuous_segments else 0,
                        'rows_before': original_len,
                        'rows_removed': removed_rows,
                        'percentage_removed': removal_percentage
                    }
                    dbscan_stats['per_file_removals'].append(file_stats)
                    
                if p_config.sliding_window:
                    segment = average_sliding_window(
                        segment, pd.Timedelta(p_config.window_size)
                    )

                output_filepath = os.path.join(
                    p_config.dir_output, output_filename
                )

                segment.to_csv(output_filepath, index=False)
                logger.info(f"Processed file saved: {output_filepath}")

    if dbscan_stats['files_processed'] > 0:
        # Create summary statistics
        summary_stats = {
            'total_files_processed': dbscan_stats['files_processed'],
            'total_rows_before': dbscan_stats['total_rows_before'],
            'total_rows_removed': dbscan_stats['total_rows_removed'],
            'total_percentage_removed': (
                dbscan_stats['total_rows_removed'] / 
                dbscan_stats['total_rows_before'] * 100
            ),
            'median_rows_removed': np.median(
                [f['rows_removed'] for f in dbscan_stats['per_file_removals']]
            )
        }
        
        # Log the summary statistics
        logger.info("\nDBSCAN Outlier Removal Summary Statistics:")
        logger.info(f"Total files processed: {summary_stats['total_files_processed']}")
        logger.info(f"Total rows before cleaning: {summary_stats['total_rows_before']}")
        logger.info(f"Total rows removed: {summary_stats['total_rows_removed']}")
        logger.info(
            f"Total percentage of points removed: {summary_stats['total_percentage_removed']:.2f}%"
        )
        logger.info(
            f"Median rows removed per file/segment: {summary_stats['median_rows_removed']}"
        )
