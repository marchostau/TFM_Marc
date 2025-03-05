import os
from enum import Enum

import pandas as pd

from .utils import concatenate_datasets
from ..logging_information.logging_config import get_logger

logger = get_logger(__name__)


class NormalizationMode(str, Enum):
    Z_STD = 'z-standardization'


def z_standardization(
        dataframe: pd.DataFrame, norm_cols: list,
        mean: pd.Series = None, std: pd.Series = None
):
    mean = dataframe[norm_cols].mean() if mean is None else mean
    std = dataframe[norm_cols].std() if std is None else std
    logger.info(
        f"Applying Z-Standardization to columns: {norm_cols} with "
        f"Mean: {mean.to_dict()}, Std Dev: {std.to_dict()}"
    )

    dataframe[norm_cols] = (
            (dataframe[norm_cols] - mean[norm_cols])
            / std[norm_cols]
        )
    return dataframe


def normalize_dataset(
        dataframe: pd.DataFrame,
        mode: NormalizationMode,
        norm_cols: list, **kwargs
):
    mode_parameters = kwargs.get("mode_parameters", {})
    logger.info(f"Normalizing dataset using mode: {mode.value}")

    match mode:
        case NormalizationMode.Z_STD:
            return z_standardization(
                    dataframe, norm_cols,
                    mean=mode_parameters.get("mean", None),
                    std=mode_parameters.get("std", None)
                )
        case _:
            logger.error("Invalid normalization mode provided")
            raise ValueError("Invalid normalization mode")


def normalize_dir_dataset(
        dir_source: str, dir_output: str,
        mode: NormalizationMode, norm_cols: list,
        daily_based: bool
):
    if not os.path.isdir(dir_source):
        logger.error(f"Invalid directory: {dir_source}")
        raise ValueError(f"'{dir_source}' is not a directory")

    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
        logger.info(f"Created output directory: {dir_output}")

    mode_parameters = {}
    if not daily_based:
        logger.info("Data normalization based on global strategy.")
        logger.info("Concatenating datasets for global normalization.")
        complete_df = concatenate_datasets(dir_source)
        if complete_df.empty:
            raise ValueError(
                "No valid data found in the source directory. "
                "Skipping normalization."
            )
        mode_parameters["mean"] = complete_df[norm_cols].mean()
        mode_parameters["std"] = complete_df[norm_cols].std()
    else:
        logger.info("Data normalization based on daily strategy.")

    files = sorted(os.listdir(dir_source))
    for filename in files:
        file_path = os.path.join(dir_source, filename)
        if os.path.isfile(file_path):
            logger.info(f"Processing file for normalization: {filename}")
            try:
                dataframe = pd.read_csv(file_path, parse_dates=["timestamp"])
            except ValueError:
                logger.warning(
                    f"Warning: CSV '{filename}' could not be read, skipping..."
                )
                continue

            normalized_df = normalize_dataset(
                dataframe, mode, norm_cols, mode_parameters=mode_parameters
            )

            output_file = os.path.join(dir_output, filename)
            normalized_df.to_csv(output_file, index=False)
            logger.info(
                f"Normalized '{filename}' and saved to '{output_file}'"
            )
