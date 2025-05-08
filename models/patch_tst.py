import os
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ray import tune
from ray.train import Checkpoint, get_checkpoint
from ray.tune import ExperimentAnalysis
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np

from transformers import PatchTSTForPrediction, PatchTSTConfig
from ..data_processing.dataset_loader import WindTimeSeriesDataset
from ..logging_information.logging_config import get_logger
from ..models.utils import (
    split_dataset, custom_collate_fn
)

logger = get_logger(__name__)


class PatchTSTWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.model_config = PatchTSTConfig(
            num_input_channels=config.get("num_input_channels", 2),
            context_length=config["lag_patch_forecast"][0],
            patch_len=config["lag_patch_forecast"][1],
            stride=config.get("patch_stride", 1),
            num_hidden_layers=config.get("num_hidden_layers", 3),
            d_model=config.get("d_model", 16),
            num_attention_heads=config.get("num_attention_heads", 4),
            share_embedding=config.get("share_embedding", True),
            channel_attention=config.get("channel_attention", False),
            ffn_dim=config.get("ffn_dim", 128),
            norm_type=config.get("norm_type", "batchnorm"),
            norm_eps=config.get("norm_eps", 1e-5),
            activation_function=config.get("activation_function", "gelu"),
            pre_norm=config.get("pre_norm", True),
            prediction_length=config["lag_patch_forecast"][2],
            target_dim=config.get("num_targets", 2),
            dropout=0.2,  # Encoder dropout
            attention_dropout=config.get("attention_dropout", 0.0),
            positional_dropout=config.get("positional_dropout", 0.0),
            path_dropout=config.get("path_dropout", 0.0),
            head_dropout=config.get("head_dropout", 0.0),
        )

        self.model = PatchTSTForPrediction(self.model_config)

    def forward(self, x):
        # x: [batch_size, input_length, input_dim]
        return self.model(past_values=x).predictions  # [batch_size, prediction_length, num_targets]        # x: [batch, context_length, target_dim]
        # x = x.permute(0, 2, 1)  # [batch, target_dim, context_length]
        # return self.model(past_values=x).logits.permute(0, 2, 1)


def build_model_config(config: dict):
    # Return just the parameters that must be passed to the patchTST model


def train_patchtst_model(config, train_ratio=0.8):
    logger.info("Starting PatchTST training...")
    logger.info(f"Training configuration: {config}")

    random_seed = config.get("random_seed", 0)
    lag, patch_size, forecast_horizon = config["lag_patch_forecast"]
    context_length = lag


    model = PatchTSTWrapper(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = getattr(optim, config["optimizer"].capitalize())(
        model.parameters(), lr=config["lr"]
    )
    criterion = nn.MSELoss()

    checkpoint = get_checkpoint()
    start_epoch = 0
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_data = torch.load(os.path.join(checkpoint_dir, "model.ckpt"))
            model.load_state_dict(checkpoint_data["model_state_dict"])
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            start_epoch = checkpoint_data["epoch"] + 1
            logger.info(f"Resumed from checkpoint at epoch {start_epoch}")

    full_dataset = WindTimeSeriesDataset(
        config["dir_source"], lag=lag,
        forecast_horizon=forecast_horizon,
        randomize=True, random_seed=random_seed
    )
    train_dataset, _ = split_dataset(full_dataset, train_ratio)

    if config.get("cap_data"):
        cap_lengths = {3: 8275, 6: 5936, 9: 4885, 12: 3924}
        max_len = cap_lengths.get(forecast_horizon, None)
        if max_len:
            train_dataset.indices = train_dataset.indices[:max_len]

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        collate_fn=custom_collate_fn
    )

    model.train()
    for epoch in range(start_epoch, config["epochs"]):
        total_loss = 0
        for batch in train_loader:
            X = batch["X"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} - Loss: {avg_loss:.6f}")

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            metrics = {"epoch": epoch, "loss": avg_loss}
            if epoch % config["checkpoint_freq"] == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss
                }, os.path.join(temp_checkpoint_dir, "model.ckpt"))
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                tune.report(metrics=metrics, checkpoint=checkpoint)
            else:
                tune.report(metrics)


def evaluate_patchtst_model(
    config,
    random_seed,
    output_dir,
    net=None,
    checkpoint_path=None,
    train_ratio=0.8
):
    logger.info("Starting PatchTST evaluation...")

    lag, forecast_horizon = config["lag_forecast"]
    if net is None:
        net = PatchTSTWrapper(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    full_dataset = WindTimeSeriesDataset(
        config["dir_source"], lag=lag,
        forecast_horizon=forecast_horizon,
        randomize=True, random_seed=random_seed
    )
    _, test_dataset = split_dataset(full_dataset, train_ratio)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        collate_fn=custom_collate_fn
    )

    net.eval()
    all_preds, all_labels, all_metadata = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            X = batch["X"].to(device)
            y = batch["y"].to(device)
            pred = net(X)

            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_metadata.append(np.stack(batch["target_metadata"], axis=0))

    preds = np.concatenate(all_preds, axis=0).reshape(-1, 2)
    labels = np.concatenate(all_labels, axis=0).reshape(-1, 2)
    metadata = np.concatenate(all_metadata, axis=0).reshape(-1, 6)

    df = pd.DataFrame({
        "true_u_component": labels[:, 0],
        "true_v_component": labels[:, 1],
        "pred_u_component": preds[:, 0],
        "pred_v_component": preds[:, 1],
        "timestamp": metadata[:, 0],
        "latitude": metadata[:, 1],
        "longitude": metadata[:, 2],
        "wind_speed": metadata[:, 3],
        "wind_direction": metadata[:, 4],
        "file_name": metadata[:, 5],
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    save_dir = os.path.join(
        output_dir,
        f"PatchTST_lag{lag}_fh{forecast_horizon}_bs{config['batch_size']}_lr{config['lr']}"
    )
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, "trues_pred_results.csv"), index=False)

    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)

    logger.info(f"PatchTST Evaluation Metrics:\nR2: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
    return {"r2": r2, "mse": mse, "mae": mae}


search_space = {
    "lag_patch_forecast": tune.grid_search([
        (3, 1, 3), (6, 1, 3), (9, 1, 3),
        (6, 1, 6), (9, 1, 6), (12, 1, 6),
        (9, 1, 9), (12, 1, 9),
        (12, 1, 12),
        (6, 2, 3), (6, 2, 6),
        (9, 3, 3), (9, 3, 6), (9, 3, 9),
        (12, 3, 6), (12, 4, 6),
        (12, 3, 9), (12, 4, 9),
        (12, 3, 12), (12, 4, 12),
        (18, 3, 3), (18, 4, 3), (18, 6, 3)
    ]),
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
    "cap_data": True,
    "random_seed": 0,

    # PatchTST-specific parameters
    "num_input_channels": 2,
    "patch_stride": 1,
    "num_hidden_layers": 3,
    "d_model": 64,
    "num_attention_heads": 4,
    "share_embedding": True,
    "channel_attention": False,
    "ffn_dim": 128,
    "norm_type": "batchnorm",
    "norm_eps": 1e-5,
    "activation_function": "gelu",
    "pre_norm": True,
    "num_targets": 2,
    "attention_dropout": 0.0,
    "positional_dropout": 0.0,
    "path_dropout": 0.0,
    "head_dropout": 0.0
}

random_seed_list = [0]
for seed in random_seed_list:
    logger.info(f"Started execution with seed: {seed}")
    storage_path = (
        '/home/marchostau/Desktop/TFM/Code/ProjectCode/models/trained_models'
    )
    search_space["random_seed"] = seed
    tune.run(
        train_patchtst_model, config=search_space, storage_path=storage_path
    )
