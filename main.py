from .data_processing.process_data import process_datasets

#config_path = '/home/marchostau/Desktop/TFM/Code/ProjectCode/config.yaml'
#process_datasets(config_path)

import numpy as np
"""
arra = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(f"{arra[4:7]}")
print(f"{arra[7:10]}")

print(f"{arra[4:7]}")
print(f"{arra[7:10]}")
print()
"""
feature_data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
lag = 3
forecast_horizon = 2
for i in range(len(feature_data) - lag - forecast_horizon):
    X = feature_data[i: i + lag]  # Input sequence
    y = feature_data[i + lag: i + lag + forecast_horizon]  # Multi-step target
    print(f"X: {X} | i: {i}")
    print(f"y: {y}")

print()
for i in range(len(feature_data) - lag - forecast_horizon + 1):
    X = feature_data[i: i + lag]  # Input sequence
    y = feature_data[i + lag: i + lag + forecast_horizon]  # Multi-step target
    print(f"X: {X} | i: {i}")
    print(f"y: {y}")
