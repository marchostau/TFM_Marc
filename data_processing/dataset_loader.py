import os
import csv
import random

import pandas as pd
import torch
from torch.utils.data import Dataset
from glob import glob

from ..logging_information.logging_config import get_logger
from ..models.utils import split_dataset

logger = get_logger(__name__)


class WindTimeSeriesDataset(Dataset):
    def __init__(
            self, dir_source: str, lag: int = 6,
            forecast_horizon: int = 1, randomize: bool = True,
            random_seed: int = 0
    ):
        self.lag = lag
        self.forecast_horizon = forecast_horizon
        self.file_list = sorted(
            glob(os.path.join(dir_source, "*.csv"))
        )

        if randomize:
            random.seed(random_seed)
            random.shuffle(self.file_list)

        logger.info(f"Randomize: {randomize} | Random seed: {random_seed}")
        logger.info(f"File list: {self.file_list}")

        self.data_indices = self._build_index()
        logger.info(f"Data indices: {self.data_indices}")

    def _build_index(self):
        logger.info("Building index of the Wind Time Series Dataset")
        logger.info(
            f"Configuration: Lag = {self.lag} | "
            f"Forecast Horizon = {self.forecast_horizon}"
        )
        dataset_info = {}
        lost_timestamps = 0
        index = []
        for file_idx, file_path in enumerate(self.file_list):
            df = pd.read_csv(
                file_path, parse_dates=["timestamp"],
                dtype={"file_name": str}, low_memory=False
            )
            df.sort_values(by=["timestamp"], inplace=True)

            feature_data = df[["u_component", "v_component"]].values

            if len(feature_data) < (self.lag + self.forecast_horizon):
                logger.warning(
                    f"Length df {len(df)} < {self.lag + self.forecast_horizon}"
                    f" | File_idx: {file_idx} | File path: {file_path}"
                )
                logger.warning(f"File {file_path} has 0 sequences")
                lost_timestamps += len(df)
                limit = self.lag + self.forecast_horizon
                dataset_info[file_path] = f"0 ({len(df)} < {limit})"
                continue

            num_sequences = 0
            if self.forecast_horizon == 1:
                for i in range(len(feature_data) - self.lag):
                    index.append((file_idx, i))
                    num_sequences += 1
            else:
                for i in range(
                    len(feature_data) - self.lag - self.forecast_horizon + 1
                ):
                    index.append((file_idx, i))
                    num_sequences += 1

            dataset_info[file_path] = num_sequences
            logger.info(f"File {file_path} has {num_sequences} of sequences")

        dataset_info["LOST TIMESTAMPS"] = lost_timestamps
        dataset_info["TOTAL"] = len(index)

        file_out_path = (
            f'/home/marchostau/Desktop/'
            f'summ_{self.lag}_{self.forecast_horizon}.csv'
        )
        with open(file_out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerows(dataset_info.items())

        logger.info(
            f"Index of the Wind Time Series Dataset built for "
            f"lag {self.lag} and forecast {self.forecast_horizon} -> "
            f"Length index: {len(index)}"
        )
        return index

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        file_idx, seq_start = self.data_indices[idx]
        file_path = self.file_list[file_idx]

        df = pd.read_csv(
            file_path, parse_dates=["timestamp"], dtype={"file_name": str}
        )
        df.sort_values(by=["timestamp"], inplace=True)

        feature_data = df[["u_component", "v_component"]].values

        if self.forecast_horizon == 1:
            X = feature_data[seq_start: seq_start + self.lag]  # Input sequence
            y = feature_data[seq_start + self.lag]  # Target
        else:
            X = feature_data[seq_start: seq_start + self.lag]  # Input sequence
            y = feature_data[seq_start + self.lag:
                             seq_start + self.lag +
                             self.forecast_horizon]  # Target

        return (torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32))
