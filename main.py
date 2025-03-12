from .data_processing.process_data import process_datasets


config_path = '/home/marchostau/Desktop/TFM/Code/ProjectCode/config.yaml'
process_datasets(config_path)


"""
import pandas as pd
from .data_processing.process_data import remove_repeated_timestamps, remove_wrong_timestamps

file_path = '/home/marchostau/Desktop/TFM/Code/ProjectCode/datasets/example_csv/2022-05-12__aut_ivan.txt.csv'
file_path = '/home/marchostau/Desktop/TFM/Code/ProjectCode/datasets/complete_datasets_csv/2019-06-06__aut_angelo.txt.csv'
dataframe = pd.read_csv(file_path, parse_dates=["timestamp"])
original_len = len(dataframe)
"""
"""
mode_timestamp = dataframe['timestamp'].dt.day.mode()[0]
print(f"Mode timestamp: {mode_timestamp} | type: {type(mode_timestamp)}")

df = dataframe[dataframe['timestamp'].dt.day == mode_timestamp]
print(f"We've passed from {original_len} rows to {len(df)}")

deleted_rows = dataframe[dataframe['timestamp'].dt.day != mode_timestamp]
print(f"Deleted rows: {deleted_rows}")

print(f"Dataframe series: {dataframe['timestamp'].dt.day}")
"""
"""
gap_threshold = pd.Timedelta('5m')
dataframe = remove_repeated_timestamps(dataframe)
dataframe = remove_wrong_timestamps(dataframe)
time_diffs = dataframe["timestamp"].diff()
print(f"Time diffs:\n{time_diffs}")

group_ids = (time_diffs > gap_threshold)
print(f"Group IDS:\n{group_ids}")

group_ids = (time_diffs > gap_threshold).cumsum()
print(f"Group IDS cumsum:\n{group_ids}")

s = dataframe.groupby(group_ids)
print(f"S:\n{s}")
print(type(s))

segments = [group for _, group in dataframe.groupby(group_ids)]
for i, segment in enumerate(segments):
    print(f"Segment {i}:\n{segment}")

print(f"Lenght segm: {len(segments)}")
"""
"""
import torch

# Creating a sample input tensor
x = torch.tensor([
    [  # First sample (batch index 0)
        [1.0, 2.0],  # Time step 1
        [3.0, 4.0],  # Time step 2
        [5.0, 6.0]   # Time step 3
    ],
    [  # Second sample (batch index 1)
        [7.0, 8.0],  # Time step 1
        [9.0, 10.0], # Time step 2
        [11.0, 12.0] # Time step 3
    ]
])

print("Original shape:", x.shape)  # (batch_size=2, seq_len=3, num_features=2)

x_flattened = x.view(2, -1)  # Flatten seq_len * num_features into one dimension

print("Flattened shape:", x_flattened.shape)  # (2, 3*2) = (2, 6)
print(x_flattened)

linear_layer = torch.nn.Linear(6, 4)  # Map from input size 6 to output size 4
out = linear_layer(x_flattened)

print("Output shape after linear layer:", out.shape)  # (2, 4)
print(out)

out_reshaped = out.view(2, -1, 2)  # Reshape back to (batch_size, output_len, num_features)

print("Reshaped output shape:", out_reshaped.shape)  # (2, 2, 2)
print(out_reshaped)
"""