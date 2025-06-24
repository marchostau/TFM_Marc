import os
from itertools import product
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ..data_processing.dataset_loader import WindTimeSeriesDataset
from ..logging_information.logging_config import get_logger
from ..models.utils import (
    split_dataset, custom_collate_fn
)

logger = get_logger(__name__)


class TransformerEncoderWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.context_length = config["context_length"]
        self.prediction_length = config["prediction_length"]
        self.num_targets = config.get("num_targets", 2)
        
        # Input embedding layer
        self.input_embedding = nn.Linear(config["num_input_channels"], config["d_model"])
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=config["d_model"],
            dropout=config.get("positional_dropout", 0.0),
            max_len=config["context_length"]
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["d_model"],
            nhead=config["num_attention_heads"],
            dim_feedforward=config["ffn_dim"],
            dropout=config.get("attention_dropout", 0.0),
            activation=config.get("activation_function", "relu"),
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.get("num_hidden_layers", 3)
        )
        
        # Output projection
        self.output_projection = nn.Linear(
            config["d_model"] * config["context_length"],
            config["prediction_length"] * config["num_targets"]
        )

    def forward(self, x):
        # x: [batch_size, context_length, num_input_channels]
        batch_size = x.size(0)
        
        # Embed input
        x = self.input_embedding(x)  # [batch_size, context_length, d_model]
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # [batch_size, context_length, d_model]
        
        # Flatten and project to output
        x = x.reshape(batch_size, -1)  # [batch_size, context_length * d_model]
        x = self.output_projection(x)  # [batch_size, prediction_length * num_targets]
        
        # Reshape to [batch_size, prediction_length, num_targets]
        return x.view(batch_size, self.prediction_length, self.num_targets)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


def train_transformer_model(
    config: dict,
    output_dir: str,
    train_ratio: float = 0.8,
    save_checkpoints: bool = True,
):
    logger.info("Starting Transformer Encoder training...")
    logger.info(f"Training configuration: {config}")
    training_start_time = time.time()

    transformer_config = {
        "num_input_channels": config.get("num_input_channels", 2),
        "context_length": config["lag_forecast"][0],
        "prediction_length": config["lag_forecast"][1],
        "num_hidden_layers": config.get("num_hidden_layers", 3),
        "d_model": config.get("d_model", 16),
        "num_attention_heads": config.get("num_attention_heads", 4),
        "ffn_dim": config.get("ffn_dim", 128),
        "activation_function": config.get("activation_function", "gelu"),
        "num_targets": config.get("num_targets", 2),
        "attention_dropout": config.get("attention_dropout", 0.0),
        "positional_dropout": config.get("positional_dropout", 0.0),
    }
    model = TransformerEncoderWrapper(transformer_config)

    early_stopping_patience = config.get("early_stopping_patience", 5)
    early_stopping_min_delta = config.get("early_stopping_min_delta", 0.0001)
    randomize = config.get("randomize", False)
    random_seed = config.get("random_seed", None)
    lag, forecast_horizon = config["lag_forecast"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = getattr(optim, config["optimizer"].capitalize())(
        model.parameters(), lr=config["lr"]
    )
    criterion = nn.MSELoss()

    full_dataset = WindTimeSeriesDataset(
        config["dir_source"], lag=lag,
        forecast_horizon=forecast_horizon,
        randomize=randomize, random_seed=random_seed
    )
    train_dataset, _ = split_dataset(full_dataset, train_ratio)

    train_num_seq = None
    cap_data = config.get("cap_data", False)
    if cap_data:
        if forecast_horizon == 3 and lag != 9:
            train_num_seq = 8275
        elif forecast_horizon == 6 and lag != 12:
            train_num_seq = 5936
        elif forecast_horizon == 9 and lag != 12:
            train_num_seq = 4885
        elif forecast_horizon == 12 and lag != 12:
            train_num_seq = 3924

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

    mean = train_data[["u_component", "v_component"]].mean()
    std = train_data[["u_component", "v_component"]].std()

    full_dataset = WindTimeSeriesDataset(
        config["dir_source"], lag=lag,
        forecast_horizon=forecast_horizon,
        randomize=randomize, random_seed=random_seed,
        mean=mean, std=std
    )
    train_dataset, _ = split_dataset(full_dataset, train_ratio)

    if train_num_seq:
        train_dataset.indices = train_dataset.indices[:train_num_seq]

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        collate_fn=custom_collate_fn,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    best_loss = float("inf")
    epochs_no_improve = 0

    history = {
        'loss': [],
        'epoch': []
    }

    model.train()
    for epoch in range(config["epochs"]):
        epoch_loss = 0.0
        for batch in train_loader:
            inputs = batch["X"].to(device)
            true_labels = batch["y"].to(device)

            optimizer.zero_grad()
            pred_labels = model(inputs)
            loss = criterion(pred_labels, true_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        history['loss'].append(avg_loss)
        history['epoch'].append(epoch)

        if avg_loss < best_loss - early_stopping_min_delta:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= early_stopping_patience:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
            break

        if save_checkpoints and (epoch % config["checkpoint_freq"] == 0):
            checkpoint_dir = os.path.join(output_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_epoch_{epoch}.pt"
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": config
            }, checkpoint_path)

            logger.info(f"Saved checkpoint to {checkpoint_path}")

    final_model_path = os.path.join(output_dir, "final_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config
    }, final_model_path)

    logger.info(f"Saved final model to {final_model_path}")
    logger.info("Training completed!")

    total_training_time = time.time() - training_start_time
    epoch = epoch + 1

    return model, history, epoch, total_training_time


def evaluate_transformer_model(
    config: dict,
    model: nn.Module,
    output_dir: str,
    random_seed: int,
    randomize: bool,
    train_ratio: float = 0.8,
):
    logger.info("Initializing Transformer Encoder model for evaluation...")
    logger.info(f"Evaluation configuration: {config}")

    lag, forecast_horizon = config["lag_forecast"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    full_dataset = WindTimeSeriesDataset(
        config["dir_source"], lag=lag,
        forecast_horizon=forecast_horizon,
        randomize=randomize, random_seed=random_seed
    )
    train_dataset, _ = split_dataset(full_dataset, train_ratio)

    train_num_seq = None
    test_num_seq = None
    cap_data = config.get("cap_data", False)
    if cap_data:
        if forecast_horizon == 3 and lag != 9:
            train_num_seq = 8275
            test_num_seq = 2069
        elif forecast_horizon == 6 and lag != 12:
            train_num_seq = 5936
            test_num_seq = 1485
        elif forecast_horizon == 9 and lag != 12:
            train_num_seq = 4885
            test_num_seq = 1222
        elif forecast_horizon == 12 and lag != 12:
            train_num_seq = 3924
            test_num_seq = 982

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

    mean = train_data[["u_component", "v_component"]].mean()
    std = train_data[["u_component", "v_component"]].std()

    full_dataset = WindTimeSeriesDataset(
        config["dir_source"], lag=lag,
        forecast_horizon=forecast_horizon,
        randomize=randomize, random_seed=random_seed,
        mean=mean, std=std
    )
    _, test_dataset = split_dataset(full_dataset, train_ratio)

    if test_num_seq:
        test_dataset.indices = test_dataset.indices[:test_num_seq]

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        collate_fn=custom_collate_fn,
        num_workers=8,
        pin_memory=True,
    )

    model.eval()
    all_preds, all_labels = [], []
    all_metadata = []
    with torch.no_grad():
        for batch in test_loader:
            inputs, true_labels = batch["X"].to(device), batch["y"].to(device)

            true_metadata = batch["target_metadata"]

            metadata_tensor = np.stack(true_metadata, axis=0)

            all_metadata.append(metadata_tensor)

            pred_labels = model(inputs)

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

    output_path = os.path.join(output_dir, "trues_pred_results.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.to_csv(output_path, index=False)

    all_preds_conc = all_preds_conc.reshape(all_preds_conc.shape[0], -1)
    all_labels_conc = all_labels_conc.reshape(all_labels_conc.shape[0], -1)

    mse = mean_squared_error(all_labels_conc, all_preds_conc)
    mae = mean_absolute_error(all_labels_conc, all_preds_conc)
    r2 = r2_score(all_labels_conc, all_preds_conc)

    logger.info(
        f"Evaluation with Transformer Encoder model Completed:\n"
        f"R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}"
    )

    return {"r2": r2, "mse": mse, "mae": mae}


def save_loss_plots(
    history: dict[str, list[float]],
    config: dict,
    plot_save_path: str
) -> None:
    model_name = "TransformerEncoder"
    batch_size = config["batch_size"]
    lag, forecast = config["lag_forecast"]
    learning_rate = config["lr"]

    plt.figure(figsize=(8, 5))
    plt.plot(
        history['epoch'], history['loss'],
        marker="o", linestyle="-", label="Loss Curve"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Loss vs. Epoch")
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


def run_experiments(
    search_space: dict,
    output_base_dir: str,
    random_seed_list: list[int] = [None],
    train_ratio: float = 0.8
) -> None:
    all_results = []

    for seed in random_seed_list:
        seed_results = []
        
        seed_dir = os.path.join(output_base_dir, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        param_names = []
        param_values = []
        for k, v in search_space.items():
            if isinstance(v, list):
                param_names.append(k)
                param_values.append(v)

        all_combinations = list(product(*param_values))

        for combination in all_combinations:
            config = search_space.copy()
            for name, value in zip(param_names, combination):
                config[name] = value

            config["random_seed"] = seed

            model_name = "TransformerEncoder"
            lag, fh = config["lag_forecast"]
            bs = config["batch_size"]
            lr = config["lr"]
            epochs = config["epochs"]

            # Transformer-specific parameters from config
            num_input_channels = config["num_input_channels"]
            num_hidden_layers = config["num_hidden_layers"]
            d_model = config["d_model"]
            num_attention_heads = config["num_attention_heads"]
            ffn_dim = config["ffn_dim"]
            activation_function = config["activation_function"]
            attention_dropout = config["attention_dropout"]
            positional_dropout = config["positional_dropout"]
            num_targets = config["num_targets"]

            exp_dir = os.path.join(
                seed_dir, model_name
            )
            exp_dir = os.path.join(
                exp_dir,
                f"lag{lag}_fh{fh}"
            )
            exp_dir = os.path.join(
                exp_dir, f"batch_size{bs}_lr{lr}"
            )
            exp_dir = os.path.join(
                exp_dir,
                f"epochs{epochs}_chs{num_input_channels}_layers{num_hidden_layers}"
                f"_dm{d_model}_heads{num_attention_heads}"
                f"_ffn{ffn_dim}_act{activation_function}"
                f"_tgt{num_targets}_attnDO{attention_dropout}_posDO{positional_dropout}"
            )
            os.makedirs(exp_dir, exist_ok=True)

            model, history, epochs, training_time = train_transformer_model(
                config=config,
                output_dir=exp_dir,
                train_ratio=train_ratio
            )

            save_loss_plots(history, config, exp_dir)

            metrics = evaluate_transformer_model(
                config=config,
                model=model,
                output_dir=exp_dir,
                random_seed=seed,
                randomize=config.get("randomize", False),
                train_ratio=train_ratio
            )

            results = {
                "random_seed": seed,
                "training_time_seconds": training_time,
                "model_class": model_name,
                "lag": lag,
                "forecast_horizon": fh,
                "batch_size": bs,
                "lr": lr,
                "dir_source": config["dir_source"],
                "optimizer": config["optimizer"],
                "epochs": epochs,
                "shuffle": config["shuffle"],
                "checkpoint_freq": config["checkpoint_freq"],
                "num_features": config["num_targets"],
                "cap_data": config["cap_data"],
                "r2": metrics["r2"],
                "mse": metrics["mse"],
                "mae": metrics["mae"],

                "num_input_channels": config["num_input_channels"],
                "num_hidden_layers": config["num_hidden_layers"],
                "d_model": config["d_model"],
                "num_attention_heads": config["num_attention_heads"],
                "ffn_dim": config["ffn_dim"],
                "activation_function": config["activation_function"],
                "num_targets": config["num_targets"],
                "attention_dropout": config["attention_dropout"],
                "positional_dropout": config["positional_dropout"],
            }
            all_results.append(results)
            seed_results.append(results)
        
        seed_results_df = pd.DataFrame(seed_results)
        seed_results_path = os.path.join(seed_dir, f"testing_results_seed{seed}.csv")
        seed_results_df.to_csv(seed_results_path, index=False)

    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(output_base_dir, "all_experiment_results.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"All experiment results saved to {results_path}")
