import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from ..data_processing.dataset_loader import WindTimeSeriesDataset, DataLoader
from ..logging_information.logging_config import get_logger
from .utils import split_dataset

logger = get_logger(__name__)


def evaluate_var_model(config, train_ratio: int = 0.8):
    logger.info("Initializing VAR model for evaluation...")
    logger.info(f"Evaluation configuration: {config}")

    lag, forecast_horizon = config["lag_forecast"]

    full_dataset = WindTimeSeriesDataset(
        config["dir_source"], lag=lag,
        forecast_horizon=forecast_horizon
    )
    train_dataset, test_dataset = split_dataset(full_dataset, train_ratio)

    file_list = []
    for idx in train_dataset.indices:
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"]
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


"""
search_space = {
    "lag_forecast_list": [
        (3, 3), (6, 6), (9, 9),
        (12, 12), (6, 3), (9, 3),
        (9, 6), (12, 6), (12, 9)
    ],
    "batch_size": 16,
    "dir_source": (
        "/home/marchostau/Desktop/TFM/Code/ProjectCode/datasets/"
        "complete_datasets_csv_processed_5m_zstd(gen)_dbscan(daily)"
    ),
    "num_features": 2,
    "shuffle": False
}

results_save_path = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/models/"
    "evaluate_results/linear_models/results_VAR[(3,3),(6,6),(9,9),(12,12),"
    "(6,3),(9,3),(9,6),(12,6),(12,9)].csv"
)

save_testing_results(search_space, results_save_path)

"""
