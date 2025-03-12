import os
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from ray import tune
from ray import train
from ray.train import Checkpoint
import pickle

from ..data_processing.dataset_loader import WindTimeSeriesDataset, DataLoader
from ..logging_information.logging_config import get_logger

logger = get_logger(__name__)


class NLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(NLinear, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        batch_size, seq_len, num_features = x.shape
        x = x.view(batch_size, -1)  # [batch_size, seq_len * num_features]
        out = self.linear(x)  # [batch_size, output_len * num_features]
        return out.view(batch_size, -1, num_features)  # [batch_size, output_len, num_features]


def split_dataset(dataset, train_ratio: int = 0.8):
    logger.info(
        f"Splitting dataset for training and "
        f"testing (train_ratio: {train_ratio})"
    )
    train_size = int(len(dataset) * train_ratio)
    indices = list(range(len(dataset)))
    logger.info(f"Train size: {train_size}")

    train_dataset = Subset(dataset, indices[:train_size])
    test_dataset = Subset(dataset, indices[train_size:])

    return train_dataset, test_dataset


def train_model(config, train_ratio: int = 0.8):
    logger.info("Initializing model for training...")
    logger.info(f"Training configuration: {config}")

    input_size = config["lag"]*2
    output_size = config["lag"]*2
    net = NLinear( 
        input_size=input_size, output_size=output_size
    )

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
    forecast_horizon = config["lag"]
    full_dataset = WindTimeSeriesDataset(
        config["dir_source"], lag=config["lag"],
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


def evaluate_model(
        config,
        net=None,
        checkpoint_path=None,
        train_ratio: int = 0.8,
        save_dir: str = None
):
    logger.info("Initializing model for evaluation...")
    logger.info(f"Evaluation configuration: {config}")

    input_size = config["lag"]*2
    output_size = config["lag"]*2
    if net is None:
        net = NLinear(
            input_size=input_size, output_size=output_size
        )

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

    forecast_horizon = config["lag"]
    full_dataset = WindTimeSeriesDataset(
        config["dir_source"], lag=config["lag"],
        forecast_horizon=forecast_horizon, train=False
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

    r2 = r2_score(all_labels_conc, all_preds_conc)
    mse = mean_squared_error(all_labels_conc, all_preds_conc)
    mae = mean_absolute_error(all_labels_conc, all_preds_conc)

    logger.info(
        f"Evaluation Completed:\n R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}"
    )

    if save_dir:
        filename = f"NLinear_lag{config['lag']}_horizon{forecast_horizon}_batch{config['batch_size']}_opt{config['optimizer']}_lr{config['lr']}_epochs{config['epochs']}.pkl"
        file_path = os.path.join(save_dir, filename)
        os.makedirs(file_path, exist_ok=True)

    with open(file_path, "wb") as f:
        pickle.dump({
            "all_preds": all_preds,
            "all_labels": all_labels,
            "r2": r2,
            "mse": mse,
            "mae": mae
        }, f)

    return {"r2": r2, "mse": mse, "mae": mae}


def train_evaluate(config):
    trained_model = train_model(config)
    results = evaluate_model(config, net=trained_model)
    return {"r2": results["r2"], "mse": results["mse"], "mae": results["mae"]}


search_space = {
    "lag": tune.grid_search([6, 9, 12]),
    #"forecast_horizon": tune.grid_search([6, 9, 12]),
    "batch_size": tune.grid_search([16, 32, 64]),
    "lr": tune.grid_search([0.001, 0.0005, 0.0001]),
    "dir_source": "/home/marchostau/Desktop/TFM/Code/ProjectCode/datasets/complete_datasets_csv_processed_5m_zstd(gen)_dbscan(daily)",
    "optimizer": "adam",
    "epochs": 250,
    "shuffle": False,
    "checkpoint_freq": 10
}
"""
search_space = {
    "lag": tune.grid_search([6]),
    #"forecast_horizon": tune.grid_search([6, 9, 12]),
    "batch_size": tune.grid_search([16]),
    "lr": tune.grid_search([0.001]),
    "dir_source": "/home/marchostau/Desktop/TFM/Code/ProjectCode/datasets/complete_datasets_csv_processed_5m_zstd(gen)_dbscan(daily)",
    "optimizer": "adam",
    "epochs": 10,
    "shuffle": False,
    "checkpoint_freq": 5
}"
"""

storage_path = '/home/marchostau/Desktop/TFM/Code/ProjectCode/models/trained_models'
tune.run(train_model, config=search_space, storage_path=storage_path)
