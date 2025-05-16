import os
from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
            context_length=config["context_length"],
            patch_len=config["patch_len"],
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
            prediction_length=config["prediction_length"],
            target_dim=config.get("num_targets", 2),
            attention_dropout=config.get("attention_dropout", 0.0),
            positional_dropout=config.get("positional_dropout", 0.0),
            path_dropout=config.get("path_dropout", 0.0),
            head_dropout=config.get("head_dropout", 0.0),
        )

        self.model = PatchTSTForPrediction(self.model_config)

    def forward(self, x):
        # x: [batch_size, input_length, input_dim]
        print(f"Input shape: {x.shape}")
        output = self.model(past_values=x).prediction_outputs  # [batch_size, prediction_length, num_targets]        # x: [batch, context_length, target_dim]
        print(f"Output shape: {output.shape}")
        return output
        # x = x.permute(0, 2, 1)  # [batch, target_dim, context_length]
        # return self.model(past_values=x).logits.permute(0, 2, 1)


def train_patchtst_model(
    config: dict,
    output_dir: str,
    train_ratio: float = 0.8,
    save_checkpoints: bool = True,
    early_stopping_patience: int = 15,
    early_stopping_min_delta: float = 0.0001
):
    #logger.info("Starting PatchTST training...")
    #logger.info(f"Training configuration: {config}")

    patchtst_config = {
        "num_input_channels": config.get("num_input_channels", 2),
        "context_length": config["lag_patch_forecast"][0],
        "patch_len": config["lag_patch_forecast"][1],
        "stride": config.get("patch_stride", 1),
        "num_hidden_layers": config.get("num_hidden_layers", 3),
        "d_model": config.get("d_model", 16),
        "num_attention_heads": config.get("num_attention_heads", 4),
        "share_embedding": config.get("share_embedding", True),
        "channel_attention": config.get("channel_attention", False),
        "ffn_dim": config.get("ffn_dim", 128),
        "norm_type": config.get("norm_type", "batchnorm"),
        "norm_eps": config.get("norm_eps", 1e-5),
        "activation_function": config.get("activation_function", "gelu"),
        "pre_norm": config.get("pre_norm", True),
        "prediction_length": config["lag_patch_forecast"][2],
        "target_dim": config.get("num_targets", 2),
        "attention_dropout": config.get("attention_dropout", 0.0),
        "positional_dropout": config.get("positional_dropout", 0.0),
        "path_dropout": config.get("path_dropout", 0.0),
        "head_dropout": config.get("head_dropout", 0.0),
    }
    model = PatchTSTWrapper(patchtst_config)

    random_seed = config.get("random_seed", 0)
    lag, _, forecast_horizon = config["lag_patch_forecast"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #logger.info(f"Using device: {device}")

    optimizer = getattr(optim, config["optimizer"].capitalize())(
        model.parameters(), lr=config["lr"]
    )
    criterion = nn.MSELoss()

    full_dataset = WindTimeSeriesDataset(
        config["dir_source"], lag=lag,
        forecast_horizon=forecast_horizon,
        randomize=True, random_seed=random_seed
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

    if train_num_seq:
        train_dataset.indices = train_dataset.indices[:train_num_seq]
        #logger.info(f"Capped data, train dataset length: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        collate_fn=custom_collate_fn,
        num_workers=8,          # Use multiple workers for data loading
        pin_memory=True,        # Faster data transfer to GPU
        persistent_workers=True # Keep workers alive between epochs
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
        #logger.info(
        #    f"Epoch {epoch + 1}/{config['epochs']} - Loss: {avg_loss:.6f}"
        #)

        history['loss'].append(avg_loss)
        history['epoch'].append(epoch)

        if avg_loss < best_loss - early_stopping_min_delta:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            #print(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= early_stopping_patience:
            #print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

        if save_checkpoints and (epoch % config["checkpoint_freq"] == 0):
            checkpoint_dir = os.path.join(output_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_epoch_{epoch}.pt"
            )
            #torch.save({
            #    "epoch": epoch,
            #    "model_state_dict": model.state_dict(),
            #    "optimizer_state_dict": optimizer.state_dict(),
            #    "loss": avg_loss,
            #    "config": config
            #}, checkpoint_path)

            #logger.info(f"Saved checkpoint to {checkpoint_path}")

    final_model_path = os.path.join(output_dir, "final_model.pt")
    #torch.save({
    #    "model_state_dict": model.state_dict(),
    #    "config": config
    #}, final_model_path)
    #logger.info(f"Saved final model to {final_model_path}")

    #logger.info("Training completed!")
    return model, history


def evaluate_patchtst_model(
    config: dict,
    model: nn.Module,
    output_dir: str,
    random_seed: int,
    randomize: bool,
    train_ratio: float = 0.8,
):
    #logger.info("Initializing linear model for evaluation...")
    #logger.info(f"Evaluation configuration: {config}")

    lag, patch_size, forecast_horizon = config["lag_patch_forecast"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #logger.info(f"Using device: {device}")

    full_dataset = WindTimeSeriesDataset(
        config["dir_source"], lag=lag,
        forecast_horizon=forecast_horizon,
        randomize=randomize, random_seed=random_seed
    )
    _, test_dataset = split_dataset(full_dataset, train_ratio)

    test_num_seq = None
    cap_data = config.get("cap_data", False)
    if cap_data:
        if forecast_horizon == 3 and lag != 9:
            test_num_seq = 2069
        elif forecast_horizon == 6 and lag != 12:
            test_num_seq = 1485
        elif forecast_horizon == 9 and lag != 12:
            test_num_seq = 1222
        elif forecast_horizon == 12 and lag != 12:
            test_num_seq = 982

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

    #logger.info(
    #    f"Evaluation with Linear models Completed:\n"
    #    f"R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}"
    #)

    return {"r2": r2, "mse": mse, "mae": mae}


def save_loss_plots(
    history: dict[str, list[float]],
    config: dict,
    plot_save_path: str
) -> None:
    #model_name = config["model_class"].__name__
    model_name = "PatchTST"
    batch_size = config["batch_size"]
    lag, patch_size, forecast = config["lag_patch_forecast"]
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
        #logger.info(f"Started execution with seed: {seed}")

        seed_dir = os.path.join(output_base_dir, f"seed_{seed}")
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

            #model_name = config["model_class"].__name__
            model_name = "PatchTST"
            lag, patch_size, fh = config["lag_patch_forecast"]
            bs = config["batch_size"]
            lr = config["lr"]

            # PatchTST-specific parameters from config
            num_input_channels = config["num_input_channels"]
            patch_stride = config["patch_stride"]
            num_hidden_layers = config["num_hidden_layers"]
            d_model = config["d_model"]
            num_attention_heads = config["num_attention_heads"]
            share_embedding = config["share_embedding"]
            channel_attention = config["channel_attention"]
            ffn_dim = config["ffn_dim"]
            norm_type = config["norm_type"]
            norm_eps = config["norm_eps"]
            activation_function = config["activation_function"]
            pre_norm = config["pre_norm"]
            num_targets = config["num_targets"]
            attention_dropout = config["attention_dropout"]
            positional_dropout = config["positional_dropout"]
            path_dropout = config["path_dropout"]
            head_dropout = config["head_dropout"]

            exp_dir = os.path.join(
                seed_dir, model_name
            )
            exp_dir = os.path.join(
                exp_dir,
                f"lag{lag}_fh{fh}"
            )
            exp_dir = os.path.join(
                exp_dir, f"patch_size{patch_size}"
            )
            exp_dir = os.path.join(
                exp_dir, f"batch_size{bs}_lr{lr}"
            )
            exp_dir = os.path.join(
                exp_dir,
                f"chs{num_input_channels}_stride{patch_stride}_layers{num_hidden_layers}_dm{d_model}_heads{num_attention_heads}"
                f"_shareEmb{int(share_embedding)}_chanAtt{int(channel_attention)}_ffn{ffn_dim}"
                f"_norm{norm_type}_eps{norm_eps}_act{activation_function}_preNorm{int(pre_norm)}"
                f"_tgt{num_targets}_attnDO{attention_dropout}_posDO{positional_dropout}"
                f"_pathDO{path_dropout}_headDO{head_dropout}"
            )
            os.makedirs(exp_dir, exist_ok=True)

            model, history = train_patchtst_model(
                config=config,
                output_dir=exp_dir,
                train_ratio=train_ratio
            )

            save_loss_plots(history, config, exp_dir)

            metrics = evaluate_patchtst_model(
                config=config,
                model=model,
                output_dir=exp_dir,
                random_seed=seed,
                randomize=config.get("randomize", False),
                train_ratio=train_ratio
            )

            results = {
                "random_seed": seed,
                "model_class": model_name,
                "lag": lag,
                "forecast_horizon": fh,
                "patch_size": patch_size,
                "batch_size": bs,
                "lr": lr,
                "dir_source": config["dir_source"],
                "optimizer": config["optimizer"],
                "epochs": config["epochs"],
                "shuffle": config["shuffle"],
                "checkpoint_freq": config["checkpoint_freq"],
                "num_features": config["num_targets"],
                "cap_data": config["cap_data"],
                "r2": metrics["r2"],
                "mse": metrics["mse"],
                "mae": metrics["mae"],

                "num_input_channels": config["num_input_channels"],
                "patch_stride": config["patch_stride"],
                "num_hidden_layers": config["num_hidden_layers"],
                "d_model": config["d_model"],
                "num_attention_heads": config["num_attention_heads"],
                "share_embedding": config["share_embedding"],
                "channel_attention": config["channel_attention"],
                "ffn_dim": config["ffn_dim"],
                "norm_type": config["norm_type"],
                "norm_eps": config["norm_eps"],
                "activation_function": config["activation_function"],
                "pre_norm": config["pre_norm"],
                "num_targets": config["num_targets"],
                "attention_dropout": config["attention_dropout"],
                "positional_dropout": config["positional_dropout"],
                "path_dropout": config["path_dropout"],
                "head_dropout": config["head_dropout"]
            }
            all_results.append(results)
            seed_results.append(results)
        
        seed_results_df = pd.DataFrame(seed_results)
        seed_results_path = os.path.join(seed_dir, f"testing_results_seed{seed}.csv")
        seed_results_df.to_csv(seed_results_path, index=False)

    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(output_base_dir, "all_experiment_results.csv")
    results_df.to_csv(results_path, index=False)
    #logger.info(f"All experiment results saved to {results_path}")


search_space = {
    "lag_patch_forecast": [
        (3, 1, 3), (6, 1, 3), (9, 1, 3),
        #(6, 1, 6), (9, 1, 6), (12, 1, 6),
        #(9, 1, 9), (12, 1, 9), (12, 1, 12),
        #(6, 2, 3), (6, 2, 6), (9, 3, 3),
        #(9, 3, 6), (9, 3, 9), (12, 3, 6),
        #(12, 4, 6), (12, 3, 9), (12, 4, 9),
        #(12, 3, 12), (12, 4, 12),
        #(18, 3, 3), (18, 4, 3), (18, 6, 3)
    ],
    #"batch_size": [16, 32, 64],
    "batch_size": [32, 64],
    "lr": [0.001, 0.0005, 0.0001],
    "dir_source": (
        "/home/nct/nct01089/Desktop/TFM/Code/ProjectCode/datasets/"
        "complete_datasets_csv_processed_5m_zstd(gen)_dbscan(daily)"
    ),
    "optimizer": "adam",
    "epochs": 150,
    "shuffle": False,
    "checkpoint_freq": 50,
    "cap_data": True,
    "random_seed": 0,

    # PatchTST-specific parameters
    "num_input_channels": 2,
    "patch_stride": 1,
    "num_hidden_layers":[3, 4],
    "d_model": [16, 128],
    "num_attention_heads": [4, 16],
    "share_embedding": True,
    "channel_attention": False,
    "ffn_dim": [128, 256],
    "norm_type": "batchnorm",
    "norm_eps": 1e-5,
    "activation_function": "gelu",
    "pre_norm": True,
    "num_targets": 2,
    "attention_dropout": [0.0, 0.2],
    "positional_dropout": 0.0,
    "path_dropout": 0.0,
    "head_dropout": 0.0
}

if __name__ == "__main__":
    output_base_dir = "/home/nct/nct01089/patchTSTResults3/patchTST_res1"
    random_seed_list = [None]  # Or [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    run_experiments(
        search_space=search_space,
        output_base_dir=os.path.join(output_base_dir, "uncapped"),
        random_seed_list=random_seed_list
    )

    search_space["cap_data"] = True
    run_experiments(
        search_space=search_space,
        output_base_dir=os.path.join(output_base_dir, "capped"),
        random_seed_list=random_seed_list
    )
