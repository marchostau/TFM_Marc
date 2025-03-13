from pydantic import BaseModel, Field
from pydantic import ValidationError
from typing import List, Optional
import yaml

from .data_cleaning import OutliersRemovalMode
from .normalization import NormalizationMode
from ..logging_information.logging_config import get_logger

logger = get_logger(__name__)


class ProcessingConfig(BaseModel):
    dir_source: str = Field(
        ..., description="Path to the input directory"
    )
    dir_output: str = Field(
        ..., description="Path to save processed datasets"
    )

    remove_duplicates: bool = Field(
        False, description="Remove repeated timestamps"
    )

    remove_wrong_dates: bool = Field(
        False, description="Remove wrong timestamps"
    )

    remove_points_outside_polygon: bool = Field(
        False, description="Remove wrong timestamps"
    )

    split_continuous_segments: bool = Field(
        False, description="Splite continous segments"
    )
    gap_threshold: str = Field(
        "5m", description="Gap threshold size (e.g., '5m' for 5 minutes)"
    )

    normalize: bool = Field(False, description="Apply normalization")
    norm_mode: Optional[NormalizationMode] = Field(
        None, description="Normalization mode"
    )
    norm_daily_based: bool = Field(
        True, description="Apply normalization with a daily based strategy"
    )
    norm_cols: List[str] = Field(
        default=[], description="Columns to normalize"
    )

    remove_outliers: bool = Field(False, description="Apply outlier removal")
    outlier_mode: Optional[OutliersRemovalMode] = Field(
        None, description="Outlier removal mode"
    )
    outlier_daily_based: bool = Field(
        True, description="Apply outlier removal with a daily based strategy"
    )
    outlier_cols: List[str] = Field(
        default=[], description="Columns to check for outliers"
    )

    sliding_window: bool = Field(
        False, description="Apply sliding window averaging"
    )
    window_size: str = Field(
        "1H", description="Window size (e.g., '1H' for 1 hour)"
    )


def load_config(config_path: str):
    with open(config_path, 'r') as file:
        raw_config = yaml.safe_load(file)

    try:
        return ProcessingConfig(**raw_config)
    except ValidationError as e:
        logger.error(f"Configuration error: {e}")
        raise