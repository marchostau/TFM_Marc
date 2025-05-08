import os

import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader

from ..data_processing.dataset_loader import WindTimeSeriesDataset
from ..logging_information.logging_config import get_logger
from .utils import (
    split_dataset,
    average_results,
    custom_collate_fn,
    get_best_results,
    obtain_pred_vs_trues_best_models,
    concatenate_all_seeds_results
)
from ..data_processing.dataset_loader import balanced_split

logger = get_logger(__name__)


def evaluate_var_model(config, output_dir: str, train_ratio: int = 0.8):
    logger.info("Initializing VAR model for evaluation...")
    logger.info(f"Evaluation configuration: {config}")

    randomize = config.get("randomize", False)
    lag, forecast_horizon = config["lag_forecast"]
    train_num_seq = config.get("train_num_seq")
    test_num_seq = config.get("test_num_seq")
    random_seed = config.get("random_seed", 0)
    train_dataset = config.get("train_dataset")
    test_dataset = config.get("test_dataset")

    #logger.info(f"Train dataset: {train_dataset}")
    #logger.info(f"Test dataset: {test_dataset}")

    if train_dataset is None and test_dataset is None:
        full_dataset = WindTimeSeriesDataset(
            config["dir_source"], lag=lag,
            forecast_horizon=forecast_horizon,
            randomize=randomize, random_seed=random_seed
        )
        train_dataset, test_dataset = split_dataset(full_dataset, train_ratio)

        file_list = []
        for idx in train_dataset.indices[:train_num_seq] if train_num_seq else train_dataset.indices:
            file_idx, seq_start = full_dataset.data_indices[idx]
            actual_file = full_dataset.file_list[file_idx]
            file_list.append((actual_file, seq_start))


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

    else:
        index_source = range(len(train_dataset)) if train_num_seq is None else range(train_num_seq)

        file_list = []
        for idx in index_source:
            file_idx, seq_start = train_dataset.data_indices[idx]
            actual_file = train_dataset.file_list[file_idx]
            file_list.append((actual_file, seq_start))

        last_file, seq_start = file_list[-1]
        file_list = list(dict.fromkeys(act_file for act_file, _ in file_list))
        final_seq = seq_start + 1

        train_data = pd.DataFrame()
        for file_path in file_list:
            act_df = pd.read_csv(file_path, parse_dates=["timestamp"])
            if file_path == last_file:
                act_df = act_df.iloc[:final_seq]
            train_data = pd.concat([train_data, act_df])

        #logger.info(f"File list training: {file_list}")
        #logger.info(f"Train data df:\n{train_data}")

    #logger.info(f"Train data indices: {train_dataset.data_indices}")
    #logger.info(f"Test data indices: {test_dataset.data_indices}")    
    #logger.info(f"Train len info: {len(train_dataset)}")
    #logger.info(f"Test len info: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        collate_fn=custom_collate_fn
    )

    model = VAR(train_data[["u_component", "v_component"]])
    model = model.fit(lag)

    all_preds, all_labels = [], []
    all_metadata = []
    for batch in test_loader:
        inputs, true_labels = batch["X"], batch["y"]

        true_metadata = batch["target_metadata"]

        metadata_tensor = np.stack(true_metadata, axis=0)

        all_metadata.append(metadata_tensor)

        for act_input, act_trues in zip(inputs, true_labels):
            pred_labels = model.forecast(act_input, steps=forecast_horizon)
            all_preds.append(pred_labels)
            all_labels.append(act_trues.numpy())

    all_preds_conc = np.concatenate(all_preds, axis=0)
    all_labels_conc = np.concatenate(all_labels, axis=0)
    all_metadata_conc = np.concatenate(all_metadata, axis=0)

    flat_labels = all_labels_conc.reshape(-1, 2)
    flat_preds = all_preds_conc.reshape(-1, 2)
    flat_metadata = all_metadata_conc.reshape(-1, 6)

    df = pd.DataFrame({
        "true_u_component": flat_labels[:, 0],
        "true_v_component": flat_labels[:, 1],
        "pred_u_component": flat_preds[:, 0],
        "pred_v_component": flat_preds[:, 1],
        "timestamp": flat_metadata[:, 0],
        "latitude": flat_metadata[:, 1],
        "longitude": flat_metadata[:, 2],
        "wind_speed": flat_metadata[:, 3],
        "wind_direction": flat_metadata[:, 4],
        "file_name": flat_metadata[:, 5],
    })

    output_path = os.path.join(
        output_dir, "model_VAR"
    )
    output_path = os.path.join(
        output_path, f"lag{lag}_fh{forecast_horizon}"
    )
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, "trues_pred_results.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.to_csv(output_path, index=False)

    mse = mean_squared_error(all_labels_conc, all_preds_conc)
    mae = mean_absolute_error(all_labels_conc, all_preds_conc)
    r2 = r2_score(all_labels_conc, all_preds_conc)

    logger.info(
        f"Evaluation with VAR model Completed:\n"
        f"R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}"
    )

    return {"r2": r2, "mse": mse, "mae": mae}


def save_testing_results(config, output_dir: str, train_ratio: int = 0.8):
    test_dates = config.get("test_dates")
    dir_source = config.get("dir_source")
    
    results = []
    for lag, forecast_horizon in config["lag_forecast_list"]:
        config["lag_forecast"] = (lag, forecast_horizon)
        if test_dates is not None:
            train_files, test_files, counts = balanced_split(
                dir_source, test_dates, lag, forecast_horizon
            )

            train_dataset = WindTimeSeriesDataset(
                dir_source=dir_source, lag=lag,
                forecast_horizon=forecast_horizon, file_list=train_files
            )
            test_dataset = WindTimeSeriesDataset(
                dir_source=dir_source, lag=lag,
                forecast_horizon=forecast_horizon, file_list=test_files
            )
            config["train_dataset"] = train_dataset
            config["test_dataset"] = test_dataset

            #logger.info(
            #    "Some testing dates are mandatory for "
            #    f"({lag}_{forecast_horizon})"
            #)

        metrics = evaluate_var_model(config, output_dir)
        metrics["config"] = (lag, forecast_horizon)
        results.append(metrics)

    random_seed = config["random_seed"]
    file_name = f"testing_results_seed{random_seed}.csv"
    output_csv_path = os.path.join(output_dir, file_name)
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    logger.info(f"Experiment results saved to {output_csv_path}")


def save_testing_results_capped(config, output_dir: str, train_ratio: int = 0.8):   
    results = []
    for lag_forecast, train_test_seq in zip(config["lag_forecast_list"], config["train_test_seqs"]):
        lag, forecast_horizon = lag_forecast
        train_num_seq, test_num_seq = train_test_seq
        config["lag_forecast"] = (lag, forecast_horizon)
        config["train_num_seq"] = train_num_seq
        config["test_num_seq"] = test_num_seq
        metrics = evaluate_var_model(config, output_dir)
        metrics["config"] = (lag, forecast_horizon)
        metrics["random_seed"] = config["random_seed"]
        results.append(metrics)

    random_seed = config["random_seed"]
    file_name = f"testing_results_seed{random_seed}.csv"
    output_csv_path = os.path.join(output_dir, file_name)
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    logger.info(f"Experiment results saved to {output_csv_path}")

"""
search_space = {
    "lag_forecast_list": [
        (3, 3), (6, 3), (9, 3),
        (6, 6), (9, 6), (12, 6),
        (9, 9), (12, 9),
        (12, 12)
    ],
    "batch_size": 16,
    "dir_source": (
        "/home/marchostau/Desktop/TFM/Code/ProjectCode/datasets/"
        "complete_datasets_csv_processed_5m_zstd(gen)_dbscan(daily)"
    ),
    "num_features": 2,
    "shuffle": False,
    "randomize": False
}

#random_seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
random_seed_list = [None]
for seed in random_seed_list:
    search_space["random_seed"] = seed
    results_save_path = (
        "/home/marchostau/Desktop/TFM/Code/ProjectCode/models/"
        "evaluate_results/var_model/results[((3,3),(6,6),"
        "(9,9),(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]_seedNone"
        f"/seed{seed}"
    )
    save_testing_results(search_space, results_save_path)


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
    "shuffle": False,
    "randomize": False
}

#random_seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
random_seed_list = [None]
for seed in random_seed_list:
    search_space["random_seed"] = seed
    results_save_path = (
        "/home/marchostau/Desktop/TFM/Code/ProjectCode/models/"
        "evaluate_results/var_model/results[((3,3),(6,6),"
        "(9,9),(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]_capped_data_seedNone"
        f"/seed{seed}"
    )
    save_testing_results_capped(search_space, results_save_path)
"""

"""
results_dir = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/var_model/results[((3,3)"
    ",(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]"
    "_seedNone/AllResults"
)
output_csv_path = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/var_model/results[((3,3)"
    ",(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]"
    "_seedNone/AllResults/results_averaged.csv"
)
#average_results(results_dir, output_csv_path, 'VAR')

output_csv_path = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/var_model/results[((3,3)"
    ",(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]"
    "_seedNone/AllResults/best_results.csv"
)
get_best_results(results_dir, output_csv_path, 'VAR')


best_results_path = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/var_model/results[((3,3)"
    ",(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]"
    "_seedNone/AllResults/best_results.csv"
)
base_results_path = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/var_model/results[((3,3)"
    ",(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]"
    "_seedNone"
)
dir_original_source = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/datasets/"
    "complete_datasets_csv"
)
base_dir_output = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/var_model/results[((3,3)"
    ",(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]"
    "_seedNone/BestResults/"
)
obtain_pred_vs_trues_best_models(
    best_results_path,
    base_results_path,
    dir_original_source,
    base_dir_output,
    model='VAR'
)
"""


"""
results_dir = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/var_model/results[((3,3)"
    ",(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]"
    "/AllResults"
)
output_csv_path = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/var_model/results[((3,3)"
    ",(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]"
    "/AllResults/concatenated_seed_results.csv"
)
concatenate_all_seeds_results(results_dir, output_csv_path)
"""

"""
search_space = {
    "lag_forecast_list": [
        (3, 3), (6, 3), (9, 3),
        (6, 6), (9, 6), (12, 6),
        (9, 9), (12, 9),
        (12, 12)
    ],
    "batch_size": 16,
    "dir_source": (
        "/home/marchostau/Desktop/TFM/Code/ProjectCode/datasets/"
        "complete_datasets_csv_processed_5m_zstd(gen)_dbscan(daily)"
    ),
    "num_features": 2,
    "shuffle": False,
    "test_dates": [
        "2024-07-29", "2024-07-16", "2024-07-13", "2024-07-07",
        "2024-06-29", "2024-06-28", "2024-06-23", "2024-06-15",
        "2024-06-13", "2024-06-03", "2024-06-03", "2024-05-30",
        "2024-06-29", "2024-06-28", "2023-10-27"
    ]
}

results_save_path = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/models/"
    "evaluate_results/var_model/results[((3,3),(6,6),"
    "(9,9),(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]_non_stationary"
)
search_space["random_seed"] = "None"
save_testing_results(search_space, results_save_path)
"""


results_dir = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/var_model/results[((3,3)"
    ",(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]"
    "_capped_data_seedNone/AllResults"
)

output_csv_path = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/var_model/results[((3,3)"
    ",(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]"
    "_capped_data_seedNone/AllResults/best_results.csv"
)
get_best_results(results_dir, output_csv_path, 'VAR')


best_results_path = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/var_model/results[((3,3)"
    ",(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]"
    "_capped_data_seedNone/AllResults/best_results.csv"
)
base_results_path = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/var_model/results[((3,3)"
    ",(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]"
    "_capped_data_seedNone"
)
dir_original_source = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/datasets/"
    "complete_datasets_csv"
)
base_dir_output = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/var_model/results[((3,3)"
    ",(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]"
    "_capped_data_seedNone/BestResults/"
)
obtain_pred_vs_trues_best_models(
    best_results_path,
    base_results_path,
    dir_original_source,
    base_dir_output,
    model='VAR'
)
