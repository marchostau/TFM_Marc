import os
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from ray import tune
from ray.train import Checkpoint, get_checkpoint
from ray.tune import ExperimentAnalysis
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from ..data_processing.dataset_loader import WindTimeSeriesDataset
from ..logging_information.logging_config import get_logger
from .utils import (
    split_dataset,
    custom_collate_fn,
    postprocess_data,
    average_results,
    get_best_results,
    obtain_pred_vs_trues_best_models,
    concatenate_all_seeds_results,
    denormalize_original_dfs
)

logger = get_logger(__name__)


class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)


class NLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(NLinear, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        # x: [Batch, Input length, Features]
        # Capture last timestep values for de-normalization
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last  # Normalize by subtracting last timestep

        # Linear layer and reshape
        x = self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = x + seq_last  # Reapply last timestep values
        return x  # [Batch, Output length, Features]


def train_linear_model(config, train_ratio: int = 0.8):
    logger.info("Initializing model for training...")
    logger.info(f"Training configuration: {config}")

    randomize = config.get("randomize", True)
    random_seed = config.get("random_seed", 0)
    lag, forecast_horizon = config["lag_forecast"]
    input_size = lag
    output_size = forecast_horizon
    model_class = config["model_class"]

    train_num_seq = config.get("train_num_seq")
    cap_data = config.get("cap_data", False)
    if cap_data:
        if forecast_horizon == 3:
            train_num_seq = 8275
        elif forecast_horizon == 6:
            train_num_seq = 5936
        elif forecast_horizon == 9:
            train_num_seq = 4885
        elif forecast_horizon == 12:
            train_num_seq = 3924

    net = model_class(input_size=input_size, output_size=output_size)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)
    logger.info(f"Using device: {device}")

    optimizer = getattr(
        optim, config["optimizer"].capitalize()
    )(net.parameters(), lr=config["lr"])
    criterion = nn.MSELoss()

    start_epoch = 0
    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_data = torch.load(os.path.join(checkpoint_dir, "model.ckpt"))
            net.load_state_dict(checkpoint_data["model_state_dict"])
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            start_epoch = checkpoint_data["epoch"] + 1
            logger.info(
                f"Resumed training from checkpoint: {checkpoint_dir}, "
                f"starting at epoch {start_epoch}"
            )

    full_dataset = WindTimeSeriesDataset(
        config["dir_source"], lag=lag,
        forecast_horizon=forecast_horizon,
        randomize=randomize, random_seed=random_seed
    )
    train_dataset, _ = split_dataset(full_dataset, train_ratio)

    if train_num_seq:
        train_dataset.indices = train_dataset.indices[:train_num_seq]
        logger.info(f"Capped data, train dataset length: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        collate_fn=custom_collate_fn
    )

    logger.info("Starting training...")
    net.train()
    for epoch in range(start_epoch, config["epochs"]):
        epoch_loss = 0.0
        for batch in train_loader:
            inputs = batch["X"].to(device)
            true_labels = batch["y"].to(device)

            optimizer.zero_grad()
            pred_labels = net(inputs)
            loss = criterion(pred_labels, true_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        logger.info(
            f"Epoch {epoch + 1}/{config['epochs']} - Loss: {avg_loss:.6f}"
        )

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            metrics = {'epoch': epoch, 'loss': avg_loss}

            if epoch % config["checkpoint_freq"] == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss
                }, os.path.join(temp_checkpoint_dir, "model.ckpt"))
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                tune.report(metrics=metrics, checkpoint=checkpoint)
            else:
                tune.report(metrics)

    logger.info("Training completed!")


def evaluate_linear_model(
        config,
        random_seed: int,
        randomize: bool,
        output_dir: str,
        net=None,
        checkpoint_path=None,
        train_ratio: int = 0.8,
):
    logger.info("Initializing linear model for evaluation...")
    logger.info(f"Evaluation configuration: {config}")

    lag, forecast_horizon = config["lag_forecast"]
    input_size = lag
    output_size = forecast_horizon
    model_name = str(config["model_class"]).split(
        '.'
    )[-1].replace("'>", "")

    if net is None:
        model_class = config["model_class"]
        net = model_class(input_size=input_size, output_size=output_size)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)
    logger.info(f"Using device: {device}")

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model checkpoint from: {checkpoint_path}")

    full_dataset = WindTimeSeriesDataset(
        config["dir_source"], lag=lag,
        forecast_horizon=forecast_horizon,
        randomize=randomize, random_seed=random_seed
    )
    _, test_dataset = split_dataset(full_dataset, train_ratio)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        collate_fn=custom_collate_fn
    )

    net.eval()
    all_preds, all_labels = [], []
    all_metadata = []
    with torch.no_grad():
        for batch in test_loader:
            inputs, true_labels = batch["X"].to(device), batch["y"].to(device)

            true_metadata = batch["target_metadata"]

            metadata_tensor = np.stack(true_metadata, axis=0)

            all_metadata.append(metadata_tensor)

            pred_labels = net(inputs)

            all_preds.append(pred_labels.cpu().numpy())
            all_labels.append(true_labels.cpu().numpy())

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

    batch_size = config["batch_size"]
    lr = config["lr"]
    output_path = os.path.join(
        output_dir, f"model_{model_name}"
    )
    output_path = os.path.join(
        output_path, f"lag{lag}_fh{forecast_horizon}"
    )
    output_path = os.path.join(
        output_path, f"batch_size{batch_size}_lr{lr}"
    )
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, "trues_pred_results.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.to_csv(output_path, index=False)

    all_preds_conc = all_preds_conc.reshape(all_preds_conc.shape[0], -1)
    all_labels_conc = all_labels_conc.reshape(all_labels_conc.shape[0], -1)

    mse = mean_squared_error(all_labels_conc, all_preds_conc)
    mae = mean_absolute_error(all_labels_conc, all_preds_conc)
    r2 = r2_score(all_labels_conc, all_preds_conc)

    logger.info(
        f"Evaluation with Linear models Completed:\n"
        f"R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}"
    )

    return {"r2": r2, "mse": mse, "mae": mae}


def save_experiment_loss_plots(experiment_path: str, plot_save_path: str):
    analysis = ExperimentAnalysis(experiment_path)

    for trial in analysis.trials:
        trial_id = trial.trial_id
        batch_size = trial.evaluated_params["batch_size"]
        lag, forecast = trial.evaluated_params["lag_forecast"]
        learning_rate = trial.evaluated_params["lr"]
        model_name = trial.config["model_class"].__name__
        df = analysis.trial_dataframes[trial_id]

        plt.figure(figsize=(8, 5))
        plt.plot(
            df["training_iteration"], df["loss"],
            marker="o", linestyle="-", label="Loss Curve"
        )

        plt.xlabel("Training Iteration")
        plt.ylabel("Loss")
        plt.title(f"{model_name} - Loss vs. Training Iteration")
        plt.legend(loc="best", fontsize="small")
        plt.grid(True)

        plot_filename = (
            f"{model_name}_batch{batch_size}_lag{lag}"
            f"_horizon{forecast}_lr{learning_rate}.png"
        )

        os.makedirs(plot_save_path, exist_ok=True)
        plot_filepath = os.path.join(plot_save_path, plot_filename)
        plt.savefig(plot_filepath, dpi=300)
        plt.close()


def save_experiment_testing_results(
        experiment_path: str, output_dir: str,
        random_seed: int, train_ratio: int = 0.8
):
    analysis = ExperimentAnalysis(experiment_path)
    results = []

    for trial in analysis.trials:
        config = trial.config
        checkpoint = trial.checkpoint

        if checkpoint:
            randomize = config["randomize"]
            model_class = config["model_class"]
            input_size = config["lag_forecast"][0]
            output_size = config["lag_forecast"][1]
            model = model_class(input_size=input_size, output_size=output_size)
            checkpoint_path = os.path.join(checkpoint.path, "model.ckpt")
            model.load_state_dict(
                torch.load(checkpoint_path)["model_state_dict"]
            )

            metrics = evaluate_linear_model(
                config, random_seed=random_seed,
                randomize=randomize, output_dir=output_dir,
                net=model, train_ratio=train_ratio
            )
            metrics["trial_id"] = trial.trial_id
            metrics["config"] = config
            results.append(metrics)

    file_name = f"testing_results_seed{random_seed}.csv"
    output_path = os.path.join(output_dir, file_name)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    logger.info(f"Experiment results saved to {output_path}")


search_space = {
    "model_class": tune.grid_search([Linear, NLinear]),
    "lag_forecast": tune.grid_search([
        (3, 3), (6, 3), (9, 3),
        (6, 6), (9, 6), (12, 6),
        (9, 9), (12, 9),
        (12, 12)
    ]
    ),
    "batch_size": tune.grid_search([16, 32, 64]),
    "lr": tune.grid_search([0.001, 0.0005, 0.0001]),
    "dir_source": (
        "/home/marchostau/Desktop/TFM/Code/ProjectCode/datasets/"
        "complete_datasets_csv_processed_5m_zstd(gen)_dbscan(daily)"
    ),
    "optimizer": "adam",
    "epochs": 70,
    "shuffle": False,
    "checkpoint_freq": 20,
    "num_features": 2,
    "cap_data": False,
    "randomize": False,
}
"""
# random_seed_list = [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
random_seed_list = [None]
for seed in random_seed_list:
    logger.info(f"Started execution with seed: {seed}")
    storage_path = (
        '/home/marchostau/Desktop/TFM/Code/ProjectCode/models/trained_models'
    )
    search_space["random_seed"] = seed
    tune.run(
        train_linear_model, config=search_space, storage_path=storage_path
    )


search_space = {
    "model_class": tune.grid_search([Linear, NLinear]),
    "lag_forecast": tune.grid_search([
        (3, 3), (6, 3), (9, 3),
        (6, 6), (9, 6), (12, 6),
        (9, 9), (12, 9),
        (12, 12)
    ]
    ),
    "batch_size": tune.grid_search([16, 32, 64]),
    "lr": tune.grid_search([0.001, 0.0005, 0.0001]),
    "dir_source": (
        "/home/marchostau/Desktop/TFM/Code/ProjectCode/datasets/"
        "complete_datasets_csv_processed_5m_zstd(gen)_dbscan(daily)"
    ),
    "optimizer": "adam",
    "epochs": 70,
    "shuffle": False,
    "checkpoint_freq": 20,
    "num_features": 2,
    "cap_data": True,
    "randomize": False,
}
"""
"""
# random_seed_list = [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
random_seed_list = [None]
for seed in random_seed_list:
    logger.info(f"Started execution with seed: {seed}")
    storage_path = (
        '/home/marchostau/Desktop/TFM/Code/ProjectCode/models/trained_models'
    )
    search_space["random_seed"] = seed
    tune.run(
        train_linear_model, config=search_space, storage_path=storage_path
    )
"""

"""
base_dir = '/home/marchostau/Desktop/TFM/Code/ProjectCode/models/trained_models'
experiment_path_list = [
    f'{base_dir}/seed0_train_linear_model_2025-04-03_20-51-26',
    f'{base_dir}/seed1_train_linear_model_2025-04-04_04-45-24',
    f'{base_dir}/seed2_train_linear_model_2025-04-04_12-19-08',
    f'{base_dir}/seed3_train_linear_model_2025-04-04_20-16-38',
    f'{base_dir}/seed4_train_linear_model_2025-04-07_20-34-38',
    f'{base_dir}/seed5_train_linear_model_2025-04-05_11-50-43',
    f'{base_dir}/seed6_train_linear_model_2025-04-05_19-24-34',
    f'{base_dir}/seed7_train_linear_model_2025-04-06_02-51-19',
    f'{base_dir}/seed8_train_linear_model_2025-04-06_10-07-40',
    f'{base_dir}/seed9_train_linear_model_2025-04-06_17-29-24',
]
random_seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

experiment_path_list = [
    f"{base_dir}/capped_data_seed0_train_linear_model_2025-04-09_23-08-28"
]
random_seed_list = [0]

for seed, experiment_path in zip(random_seed_list, experiment_path_list):
    search_space["random_seed"] = seed

    plot_save_path = (
        "/home/marchostau/Desktop/TFM/Code/ProjectCode/models/plots/"
        "loss_plots/linear_models/[(3,3),(6,6),(9,9),(12,12)"
        ",(6,3),(9,3),(9,6),(12,6),(12,9)]_capped_data/"
        f"seed{seed}"
    )

    results_save_dir = (
        "/home/marchostau/Desktop/TFM/Code/ProjectCode/models/"
        "evaluate_results/linear_models/results[((3,3),(6,6),"
        "(9,9),(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]_capped_data"
        f"/seed{seed}"
    )

    save_experiment_loss_plots(experiment_path, plot_save_path)
    save_experiment_testing_results(
        experiment_path, results_save_dir, random_seed=seed
    )
"""

"""
results_dir = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/linear_models/results"
    "[((3,3),(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),"
    "(12,6),(12,9)]/AllResults"
)
output_csv_path = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/linear_models/results"
    "[((3,3),(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),"
    "(12,6),(12,9)]/AllResults/results_averaged.csv"
)
average_results(results_dir, output_csv_path)

output_csv_path = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/linear_models/results"
    "[((3,3),(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),"
    "(12,6),(12,9)]/AllResults/best_results.csv"
)
#get_best_results(results_dir, output_csv_path)


best_results_path = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/models/"
    "evaluate_results/linear_models/results[((3,3),(6,6),"
    "(9,9),(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]"
    "/AllResults/best_results.csv"
)
base_results_path = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/linear_models/results"
    "[((3,3),(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),"
    "(12,6),(12,9)]/"
)
dir_original_source = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/datasets/"
    "complete_datasets_csv"
)
base_dir_output = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/linear_models/results"
    "[((3,3),(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),"
    "(12,6),(12,9)]/BestResults/"
)
#obtain_pred_vs_trues_best_models(
#    best_results_path,
#    base_results_path,
#    dir_original_source,
#    base_dir_output
#)

results_dir = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/linear_models/results"
    "[((3,3),(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),"
    "(12,6),(12,9)]/AllResults"
)
output_csv_path = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/linear_models/results"
    "[((3,3),(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),"
    "(12,6),(12,9)]/AllResults/"
    "concatenated_seed_results.csv"
)
concatenate_all_seeds_results(results_dir, output_csv_path)
"""


"""
dir_denorm = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/datasets/"
    "complete_datasets_csv_processed_5m_zstd(gen)_dbscan(daily)"

)
dir_original_source = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/datasets/"
    "complete_datasets_csv"
)
dir_output = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/datasets/"
    "Denormalized/complete_datasets_csv_processed_5m_zstd(gen)_dbscan(daily)"
)

denormalize_original_dfs(dir_denorm, dir_original_source, dir_output)
"""


results_dir = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/linear_models/results"
    "[((3,3),(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),"
    "(12,6),(12,9)]_capped_data/AllResults"
)
output_csv_path = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/linear_models/results"
    "[((3,3),(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),"
    "(12,6),(12,9)]_capped_data/AllResults/results_averaged.csv"
)
average_results(results_dir, output_csv_path)

output_csv_path = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/linear_models/results"
    "[((3,3),(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),"
    "(12,6),(12,9)]_capped_data/AllResults/best_results.csv"
)
get_best_results(results_dir, output_csv_path)


best_results_path = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/models/"
    "evaluate_results/linear_models/results[((3,3),(6,6),"
    "(9,9),(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]_capped_data"
    "/AllResults/best_results.csv"
)
base_results_path = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/linear_models/results"
    "[((3,3),(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),"
    "(12,6),(12,9)]_capped_data/"
)
dir_original_source = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/datasets/"
    "complete_datasets_csv"
)
base_dir_output = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/linear_models/results"
    "[((3,3),(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),"
    "(12,6),(12,9)]_capped_data/BestResults/"
)
obtain_pred_vs_trues_best_models(
    best_results_path,
    base_results_path,
    dir_original_source,
    base_dir_output
)

results_dir = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/linear_models/results"
    "[((3,3),(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),"
    "(12,6),(12,9)]_capped_data/AllResults"
)
output_csv_path = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/evaluate_results/linear_models/results"
    "[((3,3),(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),"
    "(12,6),(12,9)]_capped_data/AllResults/"
    "concatenated_seed_results.csv"
)
concatenate_all_seeds_results(results_dir, output_csv_path)
