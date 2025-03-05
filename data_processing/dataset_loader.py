import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob

class WindTimeSeriesDataset(Dataset):
    def __init__(self, dir_source: str, lag: int = 6):
        self.lag = lag
        self.file_list = sorted(glob(os.path.join(dir_source, "*.csv")))  # Store file paths
        self.data_indices = self._build_index()

    def _build_index(self):
        """Indexes file sequences to avoid storing full data in memory."""
        index = []
        for file_idx, file_path in enumerate(self.file_list):
            df = pd.read_csv(file_path, parse_dates=["timestamp"], dtype={"file_name": str}, low_memory=False)
            df.sort_values(by=["timestamp"], inplace=True)

            feature_data = df[["u_component", "v_component"]].values
            for i in range(len(feature_data) - self.lag):
                index.append((file_idx, i))
        return index

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        """Dynamically loads sequences instead of keeping everything in RAM."""
        file_idx, seq_start = self.data_indices[idx]
        file_path = self.file_list[file_idx]

        df = pd.read_csv(file_path, parse_dates=["timestamp"], dtype={"file_name": str}, low_memory=False)
        df.sort_values(by=["timestamp"], inplace=True)

        feature_data = df[["u_component", "v_component"]].values

        X = feature_data[seq_start : seq_start + self.lag]  # Input sequence
        y = feature_data[seq_start + self.lag]  # Target

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


dir_source = "/home/marchostau/Desktop/TFM/Code/ProjectCode/datasets/complete_datasets_csv_processed_10m_zstd_dbscan"
lag = 6
batch_size = 32

dataset = WindTimeSeriesDataset(dir_source, lag=lag)
#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)  # Enable parallel loading
print(f"Total number of sequences: {len(dataset)}")
print(f"Dataset:\n{dataset}")

#for X_batch, y_batch in dataloader:
#    print("Input shape:", X_batch.shape)  # (batch_size, lag, num_features)
#    print("Target shape:", y_batch.shape)  # (batch_size, num_features)