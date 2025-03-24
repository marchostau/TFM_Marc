import os
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from ray import tune
from ray.train import Checkpoint
from ray.tune import ExperimentAnalysis
import matplotlib.pyplot as plt

from ..data_processing.dataset_loader import WindTimeSeriesDataset, DataLoader
from ..logging_information.logging_config import get_logger
from .utils import split_dataset

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

    lag, forecast_horizon = config["lag_forecast"]
    input_size = lag
    output_size = forecast_horizon
    model_class = config["model_class"]
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
    """
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint = torch.load(os.path.join(checkpoint_dir, "model.ckpt"))
            net.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            logger.info(
                f"Resumed training from checkpoint: {checkpoint_dir}, "
                f"starting at epoch {start_epoch}"
            )
    """
    full_dataset = WindTimeSeriesDataset(
        config["dir_source"], lag=lag,
        forecast_horizon=forecast_horizon
    )
    train_dataset, _ = split_dataset(full_dataset, train_ratio)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"]
    )

    logger.info("Starting training...")
    net.train()
    for epoch in range(start_epoch, config["epochs"]):
        epoch_loss = 0.0
        for inputs, true_labels in train_loader:
            inputs = inputs.to(device)
            true_labels = true_labels.to(device)

            optimizer.zero_grad()
            pred_labels = net(inputs)
            loss = criterion(pred_labels, true_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            logger.info(f"Inputs:\n{inputs}")
            logger.info(f"True labels:\n{true_labels}")
            logger.info(f"Pred labels:\n{pred_labels}")
            logger.info(f"Epoch loss: {epoch_loss}")

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
        net=None,
        checkpoint_path=None,
        train_ratio: int = 0.8,
):
    logger.info("Initializing linear model for evaluation...")
    logger.info(f"Evaluation configuration: {config}")

    lag, forecast_horizon = config["lag_forecast"]
    input_size = lag
    output_size = forecast_horizon
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
        forecast_horizon=forecast_horizon
    )
    _, test_dataset = split_dataset(full_dataset, train_ratio)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"]
    )

    net.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, true_labels in test_loader:
            inputs, true_labels = inputs.to(device), true_labels.to(device)

            pred_labels = net(inputs)

            all_preds.append(pred_labels.cpu().numpy())
            all_labels.append(true_labels.cpu().numpy())

    all_preds_conc = np.concatenate(all_preds, axis=0)
    all_labels_conc = np.concatenate(all_labels, axis=0)

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

        plot_filepath = os.path.join(plot_save_path, plot_filename)
        plt.savefig(plot_filepath, dpi=300)
        plt.close()


def save_experiment_testing_results(
        experiment_path: str, output_csv_path: str, train_ratio: int = 0.8
):
    analysis = ExperimentAnalysis(experiment_path)
    results = []

    for trial in analysis.trials:
        config = trial.config
        checkpoint = trial.checkpoint

        if checkpoint:
            model_class = config["model_class"]
            input_size = config["lag_forecast"][0]
            output_size = config["lag_forecast"][1]
            model = model_class(input_size=input_size, output_size=output_size)
            checkpoint_path = os.path.join(checkpoint.path, "model.ckpt")
            model.load_state_dict(
                torch.load(checkpoint_path)["model_state_dict"]
            )

            metrics = evaluate_linear_model(
                config, net=model, train_ratio=train_ratio
            )
            metrics["trial_id"] = trial.trial_id
            metrics["config"] = config
            results.append(metrics)

    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    logger.info(f"Experiment results saved to {output_csv_path}")


"""
search_space = {
    "model_class": tune.grid_search([Linear, NLinear]),
    "lag_forecast": tune.grid_search([
        (3, 3), (6, 6), (9, 9),
        (12, 12), (6, 3), (9, 3),
        (9, 6), (12, 6), (12, 9)
    ]
    ),
    "batch_size": tune.grid_search([16, 32, 64]),
    "lr": tune.grid_search([0.001, 0.0005, 0.0001]),
    "dir_source": (
        "/home/marchostau/Desktop/TFM/Code/ProjectCode/datasets/"
        "complete_datasets_csv_processed_5m_zstd(gen)_dbscan(daily)"
    ),
    "optimizer": "adam",
    "epochs": 100,
    "shuffle": False,
    "checkpoint_freq": 25,
    "num_features": 2
}


# storage_path = '/home/marchostau/Desktop/TFM/Code/ProjectCode/models/'
# 'trained_models'
# tune.run(train_linear_model, config=search_space,
#          storage_path=storage_path)

experiment_path = (
    '/home/marchostau/Desktop/TFM/Code/ProjectCode/models/trained_models/'
    'train_model_2025-03-19_19-10-39'
)

plot_save_path = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/models/plots/loss_plots/"
    "linear_models/WithPointsOutsidePortRemoval[(3,3),(6,6),(9,9),(12,12),"
    "(6,3),(9,3),(9,6),(12,6),(12,9)]"
)

results_save_path = (
    '/home/marchostau/Desktop/TFM/Code/ProjectCode/models/'
    'evaluate_results/linear_models/results[(3,3),(6,6),(9,9),(12,12),'
    '(6,3),(9,3),(9,6),(12,6),(12,9)].csv'
)

save_experiment_loss_plots(experiment_path, plot_save_path)
save_experiment_testing_results(experiment_path, results_save_path)

"""
