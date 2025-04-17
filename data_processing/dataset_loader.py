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
            random_seed: int = 0, file_list: list = None
    ):
        self.lag = lag
        self.forecast_horizon = forecast_horizon

        if file_list is not None:
            self.file_list = sorted(file_list)
        else:
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
        file_name = os.path.basename(file_path)

        df = pd.read_csv(
            file_path, parse_dates=["timestamp"], dtype={"file_name": str}
        )
        df.sort_values(by=["timestamp"], inplace=True)

        feature_data = df[["u_component", "v_component"]].values

        if self.forecast_horizon == 1:
            X = feature_data[seq_start: seq_start + self.lag]
            y = feature_data[seq_start + self.lag]

            target_metadata_df = df[
                [
                    'timestamp', 'latitude', 'longitude',
                    'wind_speed', 'wind_direction',
                    'file_name'
                ]
            ].iloc[seq_start + self.lag].copy()

            target_metadata_df['file_name'] = file_name
            target_metadata = target_metadata_df
        else:
            X = feature_data[seq_start: seq_start + self.lag]
            y = feature_data[
                seq_start + self.lag:
                seq_start + self.lag + self.forecast_horizon
            ]

            target_metadata_df = df[
                [
                    'timestamp', 'latitude', 'longitude',
                    'wind_speed', 'wind_direction',
                    'file_name'
                ]
            ].iloc[
                seq_start + self.lag:
                seq_start + self.lag + self.forecast_horizon
            ].copy()

            target_metadata_df['file_name'] = file_name
            target_metadata = target_metadata_df.values

        input_metadata_df = df[
            [
                'timestamp', 'latitude', 'longitude',
                'wind_speed', 'wind_direction',
                'file_name'
            ]
        ].iloc[seq_start: seq_start + self.lag].copy()

        input_metadata_df['file_name'] = file_name
        input_metadata = input_metadata_df.values

        return {
            "X": torch.tensor(X, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "input_metadata": input_metadata,
            "target_metadata": target_metadata
        }


def count_sequences(file_path, lag, forecast_horizon):
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    feature_data = df[["u_component", "v_component"]].values
    if len(feature_data) < (lag + forecast_horizon):
        return 0
    if forecast_horizon == 1:
        return len(feature_data) - lag
    else:
        return len(feature_data) - lag - forecast_horizon + 1


def balanced_split(dir_source, test_dates, lag, forecast_horizon, target_ratio=0.2):
    all_files = sorted(glob(os.path.join(dir_source, "*.csv")))

    file_sequence_counts = {
        f: count_sequences(f, lag, forecast_horizon)
        for f in all_files
    }

    total_sequences = sum(file_sequence_counts.values())
    target_test_sequences = total_sequences * target_ratio

    test_files = [
        f for f in all_files if any(date in os.path.basename(f) for date in test_dates)
    ]

    current_test_sequences = sum(file_sequence_counts[f] for f in test_files)

    remaining_files = [f for f in all_files if f not in test_files]

    random.seed(42) 
    random.shuffle(remaining_files)

    for f in remaining_files:
        if current_test_sequences >= target_test_sequences:
            break
        test_files.append(f)
        current_test_sequences += file_sequence_counts[f]

    train_files = [f for f in all_files if f not in test_files]

    return train_files, test_files, file_sequence_counts


"""
dir_source = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/datasets/"
    "complete_datasets_csv_processed_5m_zstd(gen)_dbscan(daily)"
)
test_dates = [
    "2024-07-29", "2024-07-16", "2024-07-13", "2024-07-07",
    "2024-06-29", "2024-06-28", "2024-06-23", "2024-06-15",
    "2024-06-13", "2024-06-03", "2024-06-03", "2024-05-30",
    "2024-06-29", "2024-06-28", "2023-10-27"
]

lag = 6
forecast = 3

train_files, test_files, counts = balanced_split(
    dir_source, test_dates, lag, forecast
)

train_dataset = WindTimeSeriesDataset(
    dir_source=dir_source, lag=lag,
    forecast_horizon=forecast, file_list=train_files
)
test_dataset = WindTimeSeriesDataset(
    dir_source=dir_source, lag=lag,
    forecast_horizon=forecast, file_list=test_files
)

train_sequences = sum(counts[f] for f in train_files)
test_sequences = sum(counts[f] for f in test_files)
total_sequences = train_sequences + test_sequences

print(f"Lag: {lag} | Forecast horizon: {forecast}")
print(f"Train Sequences: {len(train_dataset)} ({train_sequences/total_sequences:.2%})")
print(f"Test Sequences: {len(test_dataset)} ({test_sequences/total_sequences:.2%})")

"""
"""
dir_source = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/datasets/"
    "complete_datasets_csv_processed_5m_zstd(gen)_dbscan(daily)"
)
all_files = sorted(glob(os.path.join(dir_source, "*.csv")))
test_dates = [
    "2024-07-29", "2024-07-16", "2024-07-13", "2024-07-07",
    "2024-06-29", "2024-06-28", "2024-06-23", "2024-06-15",
    "2024-06-13", "2024-06-03", "2024-06-03", "2024-05-30",
    "2024-06-29", "2024-06-28", "2023-10-27"
]
test_files = [f for f in all_files if any(date in os.path.basename(f) for date in test_dates)]
train_files = list(set(all_files) - set(test_files))

print(f"Train files: {train_files}")
print(f"Test files: {test_files}")

lag = 6
forecast = 3

train_dataset = WindTimeSeriesDataset(
    dir_source=dir_source, lag=lag,
    forecast_horizon=forecast, file_list=train_files
)
test_dataset = WindTimeSeriesDataset(
    dir_source=dir_source, lag=lag,
    forecast_horizon=forecast, file_list=test_files
)

print(f"Train dataset: {train_dataset}")
print(f"Train dataset len: {len(train_dataset)}")
print(f"Test dataset: {test_dataset}")
print(f"Test dataset len: {len(test_dataset)}")
"""