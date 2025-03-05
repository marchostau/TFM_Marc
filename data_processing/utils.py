import os

import pandas as pd

from ..logging_information.logging_config import get_logger

logger = get_logger(__name__)


def concatenate_datasets(dir_source: str):
    if not os.path.isdir(dir_source):
        raise ValueError(f'{dir_source} is not a directory')

    complete_df = pd.DataFrame()
    files = sorted(os.listdir(dir_source))
    for filename in files:
        file_path = os.path.join(dir_source, filename)
        if os.path.isfile(file_path):
            try:
                act_df = pd.read_csv(file_path, parse_dates=["timestamp"])
            except ValueError:
                logger.info(f"CSV {filename} could not be read")
                continue
            complete_df = pd.concat([complete_df, act_df])

    return complete_df
