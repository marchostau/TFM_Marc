import os
import random

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
from glob import glob


class WindTimeSeriesDataset(Dataset):
    def __init__(
            self, dir_source: str, lag: int = 6,
            forecast_horizon: int = 1, randomize: bool = False,
            random_seed: int = 0, file_list: list = None,
            mean: torch.Tensor = None, std: torch.Tensor = None
    ):
        self.lag = lag
        self.forecast_horizon = forecast_horizon
        self.mean = mean
        self.std = std

        if file_list is not None:
            self.file_list = sorted(file_list)
        else:
            self.file_list = sorted(
                glob(os.path.join(dir_source, "*.csv"))
            )
            if random_seed is not None and str(random_seed).lower() != "none":
                random_seed = int(random_seed)

        self.data_indices = self._build_index()

    def _build_index(self):
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

        dataset_info["LOST TIMESTAMPS"] = lost_timestamps
        dataset_info["TOTAL SEQUENCES"] = len(index)

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

        if self.mean is not None and self.std is not None:
            X = (X - self.mean.to_numpy()) / self.std.to_numpy()
            y = (y - self.mean.to_numpy()) / self.std.to_numpy()

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


def analyze_segment_balance(dir_source, lag_forecast_list, train_ratio=0.8, randomize=True, random_seed=0):
    results = []

    for lag, forecast in lag_forecast_list:
        full_dataset = WindTimeSeriesDataset(
            dir_source=dir_source,
            lag=lag,
            forecast_horizon=forecast,
            randomize=randomize,
            random_seed=random_seed
        )

        train_dataset, test_dataset = split_dataset(full_dataset, train_ratio)

        train_indices = train_dataset.indices
        test_indices = test_dataset.indices

        total_segment_ids = set([segment_num for segment_num,_ in full_dataset.data_indices])
        train_file_ids = set([full_dataset.data_indices[i][0] for i in train_indices])
        test_file_ids = set([full_dataset.data_indices[i][0] for i in test_indices])

        train_percentage = len(train_file_ids) / len(total_segment_ids) * 100
        test_percentage = (len(test_file_ids)-1) / (len(total_segment_ids))* 100

        overlap_ids = train_file_ids & test_file_ids
        overlap_count = len(overlap_ids)

        results.append({
            'lag': lag,
            'forecast': forecast,
            'num_train_segments': len(train_file_ids),
            'num_test_segments': len(test_file_ids),
            'train_percentage': round(train_percentage, 2),
            'test_percentage': round(test_percentage, 2),
            'overlap_ids': overlap_ids,
            'overlap_count': overlap_count
        })

    return pd.DataFrame(results)


def split_dataset(dataset, train_ratio: int = 0.8):
    train_size = int(len(dataset) * train_ratio)
    indices = list(range(len(dataset)))

    train_dataset = Subset(dataset, indices[:train_size])
    test_dataset = Subset(dataset, indices[train_size:])

    return train_dataset, test_dataset


def obtain_train_data(
        dir_source: str,
        lag: int,
        forecast_horizon: int,
        random_seed: int,
        train_ratio: float = 0.8,
        train_num_seq: int = None
):
    full_dataset = WindTimeSeriesDataset(
        dir_source, lag=lag,
        forecast_horizon=forecast_horizon,
        randomize=False, random_seed=random_seed
    )
    train_dataset, _ = split_dataset(full_dataset, train_ratio)

    file_list = []
    for idx in train_dataset.indices[:train_num_seq] if train_num_seq else train_dataset.indices:
        file_idx, seq_start = full_dataset.data_indices[idx]
        actual_file = full_dataset.file_list[file_idx]
        file_list.append((actual_file, seq_start))

    last_file, seq_start = file_list[-1]
    file_list = list(dict.fromkeys(act_file for act_file, _ in file_list))
    final_seq = seq_start + 1

    file_list = sorted(file_list)

    train_data = pd.DataFrame()
    for file_path in file_list:
        act_df = pd.read_csv(file_path, parse_dates=["timestamp"])
        if file_path == last_file:
            act_df = act_df.iloc[:final_seq]
        train_data = pd.concat([train_data, act_df])

    return train_data

