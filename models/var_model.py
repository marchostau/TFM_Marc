import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader

from ..data_processing.dataset_loader import WindTimeSeriesDataset
from ..logging_information.logging_config import get_logger
from .utils import split_dataset, average_results, custom_collate_fn

logger = get_logger(__name__)


def evaluate_var_model(config, train_ratio: int = 0.8):
    logger.info("Initializing VAR model for evaluation...")
    logger.info(f"Evaluation configuration: {config}")

    lag, forecast_horizon = config["lag_forecast"]
    train_num_seq = config.get("train_num_seq")
    test_num_seq = config.get("test_num_seq")
    random_seed = config.get("random_seed", 0)

    full_dataset = WindTimeSeriesDataset(
        config["dir_source"], lag=lag,
        forecast_horizon=forecast_horizon,
        randomize=True, random_seed=random_seed
    )
    train_dataset, test_dataset = split_dataset(full_dataset, train_ratio)

    print(f"Lag: {lag} | Forecast horizon: {forecast_horizon}")
    print(f"Initial train data length: {len(train_dataset)}")
    print(f"Initial test data length: {len(test_dataset)}")

    file_list = []
    for idx in train_dataset.indices[:train_num_seq] if train_num_seq else train_dataset.indices:
        file_idx, seq_start = full_dataset.data_indices[idx]
        actual_file = full_dataset.file_list[file_idx]
        file_list.append((actual_file, seq_start))

    print(f"Capped train data length: {len(file_list)}")

    last_file, seq_start = file_list[-1]
    file_list = list(dict.fromkeys(act_file for act_file, _ in file_list))
    final_seq = seq_start + 1

    train_data = pd.DataFrame()
    for file_path in file_list:
        act_df = pd.read_csv(file_path, parse_dates=["timestamp"])
        if file_path == last_file:
            act_df = act_df.iloc[:final_seq]
        train_data = pd.concat([train_data, act_df])

    if test_num_seq:
        test_dataset.indices = test_dataset.indices[:test_num_seq]

    print(f"Capped test data length: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        collate_fn=custom_collate_fn
    )

    model = VAR(train_data[["u_component", "v_component"]])
    model = model.fit(lag)

    all_preds, all_labels = [], []
    for inputs, true_labels in test_loader:
        for act_input, act_trues in zip(inputs, true_labels):
            pred_labels = model.forecast(act_input, steps=forecast_horizon)
            all_preds.append(pred_labels)
            all_labels.append(act_trues.numpy())

    all_preds_conc = np.concatenate(all_preds, axis=0)
    all_labels_conc = np.concatenate(all_labels, axis=0)

    mse = mean_squared_error(all_labels_conc, all_preds_conc)
    mae = mean_absolute_error(all_labels_conc, all_preds_conc)
    r2 = r2_score(all_labels_conc, all_preds_conc)

    logger.info(
        f"Evaluation with VAR model Completed:\n"
        f"R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}"
    )

    return {"r2": r2, "mse": mse, "mae": mae}


def save_testing_results(config, output_csv_path: str, train_ratio: int = 0.8):
    results = []
    for lag, forecast_horizon in config["lag_forecast_list"]:
        config["lag_forecast"] = (lag, forecast_horizon)
        metrics = evaluate_var_model(config)
        metrics["config"] = (lag, forecast_horizon)
        results.append(metrics)

    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    logger.info(f"Experiment results saved to {output_csv_path}")


def save_testing_results_capped(config, output_csv_path: str, train_ratio: int = 0.8):   
    results = []
    for lag_forecast, train_test_seq in zip(config["lag_forecast_list"], config["train_test_seqs"]):
        lag, forecast_horizon = lag_forecast
        train_num_seq, test_num_seq = train_test_seq
        config["lag_forecast"] = (lag, forecast_horizon)
        config["train_num_seq"] = train_num_seq
        config["test_num_seq"] = test_num_seq
        metrics = evaluate_var_model(config)
        metrics["config"] = (lag, forecast_horizon)
        results.append(metrics)

    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    logger.info(f"Experiment results saved to {output_csv_path}")


search_space = {
    "lag_forecast_list": [
        (3, 3), (6, 3), (9, 3),
        (6, 6), (9, 6), (12, 6),
        (9, 9), (12, 9),
        (12, 12)
    ],
    "train_test_seqs": [
        (8275, 2069), (8275, 2069), (8275, 2069),
        (5936, 1485), (5936, 1485), (5936, 1485),
        (4885, 1222), (4885, 1222),
        (3924, 982)
    ],
    "batch_size": 16,
    "dir_source": (
        "/home/marchostau/Desktop/TFM/Code/ProjectCode/datasets/"
        "complete_datasets_csv_processed_5m_zstd(gen)_dbscan(daily)"
    ),
    "num_features": 2,
    "shuffle": False
}

"""
random_seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for seed in random_seed_list:
    search_space["random_seed"] = seed
    results_save_path = (
        "/home/marchostau/Desktop/TFM/Code/ProjectCode/models/"
        "evaluate_results/var_model/diff_seeds_capped/results_VAR[(3,3),"
        f"(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]_seed{seed}.csv"
    )
    #save_testing_results(search_space, results_save_path)
    save_testing_results_capped(search_space, results_save_path)

"""
results_dir = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/var_model/diff_seeds_capped"
)
output_csv_path = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/var_model/diff_seeds_capped/results_averaged.csv"
)
average_results(results_dir, output_csv_path)
