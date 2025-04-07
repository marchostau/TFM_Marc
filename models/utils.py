import os
from glob import glob
import pandas as pd
import torch
from torch.utils.data import Subset

from ..logging_information.logging_config import get_logger

logger = get_logger(__name__)


def split_dataset(dataset, train_ratio: int = 0.8):
    logger.info(
        f"Splitting dataset for training and "
        f"testing (train_ratio: {train_ratio})"
    )
    train_size = int(len(dataset) * train_ratio)
    indices = list(range(len(dataset)))
    logger.info(f"Train size: {train_size}")

    train_dataset = Subset(dataset, indices[:train_size])
    test_dataset = Subset(dataset, indices[train_size:])

    return train_dataset, test_dataset


def average_results(results_dir: list, output_csv_path: str):
    file_list = sorted(
        glob(os.path.join(results_dir, "*.csv"))
    )

    conc_df = pd.DataFrame()
    for file_path in file_list:
        act_df = pd.read_csv(file_path)
        conc_df = pd.concat([conc_df, act_df])

    grouped_df = conc_df.groupby("config").mean()
    grouped_df = grouped_df.reset_index()
    grouped_df.to_csv(output_csv_path, index=False)


def custom_collate_fn(batch):
    batch_dict = {
        "X": torch.stack([item["X"] for item in batch]),
        "y": torch.stack([item["y"] for item in batch]),
        "input_metadata": [item["input_metadata"] for item in batch],
        "target_metadata": [item["target_metadata"] for item in batch],
    }
    return batch_dict
