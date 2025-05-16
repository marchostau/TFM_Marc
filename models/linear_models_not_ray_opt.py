import os
import tempfile
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from ..data_processing.dataset_loader import WindTimeSeriesDataset
from ..logging_information.logging_config import get_logger
from .utils import (
    split_dataset,
    custom_collate_fn,
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


def train_linear_model(
    config: dict,
    output_dir: str,
    train_ratio: float = 0.8,
    save_checkpoints: bool = True
):
    logger.info("Initializing model for training...")
    logger.info(f"Training configuration: {config}")

    randomize = config.get("randomize", False)
    random_seed = config.get("random_seed", 0)
    lag, forecast_horizon = config["lag_forecast"]
    input_size = lag
    output_size = forecast_horizon
    model_class = config["model_class"]

    net = model_class(input_size=input_size, output_size=output_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    logger.info(f"Using device: {device}")

    optimizer = getattr(
        optim, config["optimizer"].capitalize()
    )(net.parameters(), lr=config["lr"])
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

    if train_num_seq:
        train_dataset.indices = train_dataset.indices[:train_num_seq]
        logger.info(f"Capped data, train dataset length: {len(train_dataset)}")

    print(f"Train dataset: {train_dataset}")
    print(f"Train dataset len: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        collate_fn=custom_collate_fn,
        num_workers=8,          # Use multiple workers for data loading
        pin_memory=True,        # Faster data transfer to GPU
        persistent_workers=True # Keep workers alive between epochs
    )

    history = {
        'loss': [],
        'epoch': []
    }

    logger.info("Starting training...")
    net.train()
    for epoch in range(config["epochs"]):
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

        history['loss'].append(avg_loss)
        history['epoch'].append(epoch)
        """
        if save_checkpoints and (epoch % config["checkpoint_freq"] == 0):
            checkpoint_dir = os.path.join(output_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_epoch_{epoch}.pt"
            )

            torch.save({
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": config
            }, checkpoint_path)

            logger.info(f"Saved checkpoint to {checkpoint_path}")
        """

    final_model_path = os.path.join(output_dir, "final_model.pt")
    torch.save({
        "model_state_dict": net.state_dict(),
        "config": config
    }, final_model_path)
    logger.info(f"Saved final model to {final_model_path}")

    logger.info("Training completed!")
    return net, history


def evaluate_linear_model(
    config: dict,
    model: nn.Module,
    output_dir: str,
    random_seed: int,
    randomize: bool,
    train_ratio: float = 0.8,
):
    logger.info("Initializing linear model for evaluation...")
    logger.info(f"Evaluation configuration: {config}")

    lag, forecast_horizon = config["lag_forecast"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")

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
        print(f"test_num_seq: {test_num_seq}")
        print(f"Original test indices: {len(test_dataset.indices)}")
        test_dataset.indices = test_dataset.indices[:test_num_seq]
        print(f"Capped test indices: {len(test_dataset.indices)}")

    print(f"Test dataset: {test_dataset}")
    print(f"Test dataset len: {len(test_dataset)}")

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
        f"Evaluation with Linear models Completed:\n"
        f"R²: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}"
    )

    return {"r2": r2, "mse": mse, "mae": mae}


def save_loss_plots(
    history: dict[str, list[float]],
    config: dict,
    plot_save_path: str
) -> None:
    model_name = config["model_class"].__name__
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
        logger.info(f"Started execution with seed: {seed}")

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

            model_name = config["model_class"].__name__
            lag, fh = config["lag_forecast"]
            bs = config["batch_size"]
            lr = config["lr"]


            cap_data = config["cap_data"]
            print(f"Lag: {lag} | Forecast: {fh} | capped_data: {cap_data}")


            exp_dir = os.path.join(
                seed_dir, f"model_{model_name}"
            )
            exp_dir = os.path.join(
                exp_dir,
                f"lag{lag}_fh{fh}"
            )
            exp_dir = os.path.join(
                exp_dir, f"batch_size{bs}_lr{lr}"
            )
            os.makedirs(exp_dir, exist_ok=True)

            model, history = train_linear_model(
                config=config,
                output_dir=exp_dir,
                train_ratio=train_ratio
            )

            save_loss_plots(history, config, exp_dir)

            metrics = evaluate_linear_model(
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
                "batch_size": bs,
                "lr": lr,
                "dir_source": config["dir_source"],
                "optimizer": config["optimizer"],
                "epochs": config["epochs"],
                "shuffle": config["shuffle"],
                "checkpoint_freq": config["checkpoint_freq"],
                "num_features": config["num_features"],
                "cap_data": config["cap_data"],
                "r2": metrics["r2"],
                "mse": metrics["mse"],
                "mae": metrics["mae"]
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


search_space = {
    "model_class": [Linear, NLinear],
    "lag_forecast": [
        (3, 3), (6, 3), (9, 3),
        (6, 6), (9, 6), (12, 6),
        (9, 9), (12, 9),
        (12, 12),
    ],
    "batch_size": [16, 32, 64],
    "lr": [0.001, 0.0005, 0.0001],
    "dir_source": (
        "/home/nct/nct01089/Desktop/TFM/Code/ProjectCode/datasets/"
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

if __name__ == "__main__":
    output_base_dir = "/home/nct/nct01089/linear_results_4"
    random_seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, None]

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
