from enum import Enum

import pandas as pd

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


def reverse_z_standardization(
        dataframe: pd.DataFrame,
        denorm_cols: dict,
        mean: pd.Series,
        std: pd.Series
):
    logger.info(
        f"Reversing Z-Standardization for columns: {denorm_cols} with "
        f"Mean: {mean.to_dict()}, Std Dev: {std.to_dict()}"
    )

    for new_col, original_col in denorm_cols.items():
        logger.debug(
            f"Reversing column: {new_col} using "
            f"original: {original_col}"
        )
        dataframe[new_col] = (
            dataframe[new_col] * std[original_col] +
            mean[original_col]
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
