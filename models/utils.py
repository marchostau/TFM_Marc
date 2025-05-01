import os
import re
import pandas as pd
import torch
from torch.utils.data import Subset

from ..logging_information.logging_config import get_logger
from ..data_processing.normalization import reverse_z_standardization
from ..data_processing.utils import concatenate_datasets
from ..data_processing.file_loader import (
    compute_wind_speed,
    compute_wind_direction
)

logger = get_logger(__name__)


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


def normalize_config(config_str: str, remove_key_str: list):
    try:
        config = config_str.strip("{}")
        config = [x.split(": ") for x in config.split(", '")]
        config = [(x[0].replace("'", ''), x[1]) for x in config]
        config = dict(config)
        for key in remove_key_str:
            config.pop(key, None)
        str_config = str(config)
        return str_config
    except (ValueError, SyntaxError):
        return config_str


def get_seed_file_list(results_dir: str):
    return sorted([
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if (f.endswith('.csv') and
            os.path.isfile(os.path.join(results_dir, f)) and
            re.search(r'seed\d+\.csv$', f))
    ])


def average_results(
        results_dir: str,
        output_csv_path: str,
        model_results: str = 'Linear'
):
    file_list = get_seed_file_list(results_dir)

    if not file_list:
        raise ValueError("No CSV files found in directory")

    conc_df = pd.concat(
        [pd.read_csv(f) for f in file_list],
        ignore_index=True
    )

    if model_results == 'Linear':
        remove_keys = ['random_seed', 'cap_data']
        conc_df['normalized_config'] = conc_df['config'].apply(
            lambda x: normalize_config(x, remove_keys)
        )

        grouped_df = conc_df.groupby('normalized_config').agg({
            'r2': 'mean',
            'mse': 'mean',
            'mae': 'mean',
        }).reset_index()

        grouped_df = grouped_df.rename(
            columns={'normalized_config': 'config'}
        )
    else:
        grouped_df = conc_df.groupby('config').agg({
            'r2': 'mean',
            'mse': 'mean',
            'mae': 'mean',
        }).reset_index()

    grouped_df.to_csv(output_csv_path, index=False)
    return grouped_df


def get_best_results(
        results_dir: str,
        output_csv_path: str,
        model_results: str = 'Linear'
):
    file_list = get_seed_file_list(results_dir)

    df_list = []
    for f in file_list:
        df = pd.read_csv(f)
        df['source_file'] = os.path.basename(f)
        df_list.append(df)

    conc_df = pd.concat(df_list, ignore_index=True)

    if model_results == 'Linear':
        remove_keys = [
            'random_seed', 'cap_data', 'batch_size', 'lr',
            'dir_source', 'optimizer', 'epochs', 'shuffle',
            'checkpoint_freq', 'num_features'
        ]
        conc_df['normalized_config'] = conc_df['config'].apply(
            lambda x: normalize_config(x, remove_keys)
        )
        best_idx = conc_df.groupby('normalized_config')['mse'].idxmin()
    else:
        best_idx = conc_df.groupby('config')['mse'].idxmin()

    best_rows = conc_df.loc[best_idx].copy()
    best_rows.to_csv(output_csv_path, index=False)

    return best_rows


def custom_collate_fn(batch):
    batch_dict = {
        "X": torch.stack([item["X"] for item in batch]),
        "y": torch.stack([item["y"] for item in batch]),
        "input_metadata": [item["input_metadata"] for item in batch],
        "target_metadata": [item["target_metadata"] for item in batch],
    }
    return batch_dict


def denormalize_original_dfs(
        dir_denorm: str,
        dir_original_source: str,
        dir_output: str,
        original_norm_cols: list = [
            "latitude", "longitude", "wind_speed",
            "wind_direction", "u_component", "v_component"
        ]
):
    df = concatenate_datasets(dir_denorm)
    original_source_df = concatenate_datasets(dir_original_source)

    print(original_source_df)

    original_mean = original_source_df[original_norm_cols].mean()
    original_std = original_source_df[original_norm_cols].std()

    denorm_cols = {
        "u_component": "u_component",
        "v_component": "v_component",
        "latitude": "latitude",
        "longitude": "longitude",
        "wind_speed": "wind_speed",
        "wind_direction": "wind_direction"
    }

    for file_name, file_df in df.groupby("file_name"):
        print(file_name)
        print(file_df)
        denorm_df = reverse_z_standardization(
            file_df,
            denorm_cols,
            original_mean,
            original_std
        )
        """
        denorm_df["wind_speed"] = denorm_df.apply(
            lambda row: compute_wind_speed(
                row['u_component'],
                row['v_component']
            ), axis=1
        )
        denorm_df["pred_wind_direction"] = denorm_df.apply(
            lambda row: compute_wind_direction(
                row['u_component'],
                row['v_component']
            ), axis=1
        )

        denorm_df["wind_speed"] = denorm_df[
            "wind_speed"
        ].apply(lambda x: float(str(x).split()[0]))
        denorm_df["wind_direction"] = denorm_df[
            "wind_direction"
        ].apply(lambda x: float(str(x).split()[0]))
        """
        logger.info(f"Df ({file_name}) after posprocess:\n{denorm_df}")

        os.makedirs(dir_output, exist_ok=True)
        file_output = os.path.join(dir_output, f"{file_name}.csv")
        denorm_df.to_csv(file_output, index=False)


def remove_segment_from_filename(file_name: str):
    return re.sub(r'_segment_\d', '', file_name)


def postprocess_data(
        df: pd.DataFrame,
        dir_original_source: str,
        dir_output: str,
        original_norm_cols: list = [
            "latitude", "longitude", "wind_speed",
            "wind_direction", "u_component", "v_component"
        ]
):
    mean_cols = ["pred_u_component", "pred_v_component"]
    group_cols = ["file_name", "timestamp"]
    constant_cols = [
        "true_u_component",
        "true_v_component",
        "latitude",
        "longitude",
        "wind_speed",
        "wind_direction"
    ]

    # Here you can try to remove _segment_X part of the file name.
    # Then you will have the segments of each day joined.
    df["file_name"] = df.apply(
        lambda row: remove_segment_from_filename(
            row["file_name"]
        ), axis=1
    )
    print(f"New DF:\n{df}")

    agg_dict = {col: "mean" for col in mean_cols}
    agg_dict.update({col: "first" for col in constant_cols})

    grouped_df = df.groupby(group_cols).agg(agg_dict).reset_index()

    original_source_df = concatenate_datasets(dir_original_source)
    original_mean = original_source_df[original_norm_cols].mean()
    original_std = original_source_df[original_norm_cols].std()

    logger.info(f"Original mean: {original_mean}")
    logger.info(f"Original std: {original_std}")

    denorm_cols = {
        "true_u_component": "u_component",
        "pred_u_component": "u_component",
        "true_v_component": "v_component",
        "pred_v_component": "v_component",
        "latitude": "latitude",
        "longitude": "longitude",
        "wind_speed": "wind_speed",
        "wind_direction": "wind_direction"
    }

    for file_name, file_df in grouped_df.groupby("file_name"):
        denorm_df = reverse_z_standardization(
            file_df,
            denorm_cols,
            original_mean,
            original_std
        )

        denorm_df.rename({
            'wind_speed': 'original_wind_speed',
            'wind_direction': 'original_wind_direction'
        })

        denorm_df["wind_speed"] = denorm_df.apply(
            lambda row: compute_wind_speed(
                row['pred_u_component'],
                row['pred_v_component']
            ), axis=1
        )
        denorm_df["wind_direction"] = denorm_df.apply(
            lambda row: compute_wind_direction(
                row['pred_u_component'],
                row['pred_v_component']
            ), axis=1
        )

        denorm_df["wind_speed"] = denorm_df[
            "wind_speed"
        ].apply(lambda x: float(str(x).split()[0]))
        denorm_df["wind_direction"] = denorm_df[
            "wind_direction"
        ].apply(lambda x: float(str(x).split()[0]))

        logger.info(f"Df ({file_name}) after posprocess:\n{denorm_df}")

        os.makedirs(dir_output, exist_ok=True)
        file_output = os.path.join(dir_output, file_name)
        denorm_df.to_csv(file_output, index=False)


def obtain_pred_vs_trues_best_models(
        best_results_path: str,
        base_results_path: str,
        dir_original_source: str,
        base_dir_output: str,
        original_norm_cols: list = [
            "latitude", "longitude", "wind_speed",
            "wind_direction", "u_component", "v_component"
        ],
        model_name: str = 'Linear'
):
    best_results_df = pd.read_csv(best_results_path)

    for idx, row in best_results_df.iterrows():
        if model_name == 'Linear':
            config = row["config"].strip("{}")
            config = [x.split(": ") for x in config.split(", '")]
            config = [(x[0].replace("'", ''), x[1]) for x in config]
            config = dict(config)

            seed = config["random_seed"]
            model_name = str(config["model_class"]).split(
                '.'
            )[-1].replace("'>", "")
            lag_forecast = config["lag_forecast"].split(",")
            lag_forecast = [value.strip("[] ").strip() for value in lag_forecast]
            lag = lag_forecast[0]
            fh = lag_forecast[1]
            batch_size = config["batch_size"]
            lr = config["lr"]

            file_path = (
                f"seed{seed}/model_{model_name}/lag{lag}_fh{fh}/"
                f"batch_size{batch_size}_lr{lr}/trues_pred_results.csv"
            )
            file_path = os.path.join(base_results_path, file_path)
            trues_pred_df = pd.read_csv(file_path)

            print(f"Processing {file_path}")
            print(f"Df obtained:\n{trues_pred_df}")

            file_output = (
                f"seed{seed}_model{model_name}_lag{lag}_fh{fh}_"
                f"batch_size{batch_size}_lr{lr}"
            )
            dir_output = os.path.join(base_dir_output, file_output)

            print(f"Going to put results in dir {dir_output}")

        elif model_name == 'VAR':
            config = row["config"]
            lag_forecast = config.split(",")
            lag_forecast = [
                value.strip(
                    "() "
                ).strip() for value in lag_forecast
            ]
            lag = lag_forecast[0]
            fh = lag_forecast[1]
            source_file = row["source_file"]
            seed = (source_file.split("_"))[2]
            seed = (seed.split("."))[0]

            file_path = (
                f"{seed}/model_VAR/lag{lag}_fh{fh}/"
                "trues_pred_results.csv"
            )

            file_path = os.path.join(base_results_path, file_path)
            trues_pred_df = pd.read_csv(file_path)

            print(f"Processing {file_path}")
            print(f"Df obtained:\n{trues_pred_df}")

            file_output = (f"{seed}_model_VAR_lag{lag}_fh{fh}")
            dir_output = os.path.join(base_dir_output, file_output)

            print(f"Going to put results in dir {dir_output}")

        postprocess_data(
            trues_pred_df,
            dir_original_source,
            dir_output,
            original_norm_cols
        )


def concatenate_all_seeds_results(results_dir: str, output_csv_path: str = None):
    file_list = get_seed_file_list(results_dir)

    if not file_list:
        raise ValueError("No CSV files found in directory")

    conc_df = pd.concat(
        [pd.read_csv(f) for f in file_list],
        ignore_index=True
    )

    if output_csv_path is not None:
        conc_df.to_csv(output_csv_path, index=False)

    return conc_df


def get_std_between_seed_results(
        conc_res_df: pd.DataFrame,
        output_csv_path: str,
        model_results: str = 'Linear'
):
    if model_results == 'Linear':
        remove_keys = ['random_seed', 'cap_data']
        conc_res_df['normalized_config'] = conc_res_df['config'].apply(
            lambda x: normalize_config(x, remove_keys)
        )

        grouped_df = conc_res_df.groupby('normalized_config').agg({
            'r2': 'std',
            'mse': 'std',
            'mae': 'std',
        }).reset_index()

        grouped_df = grouped_df.rename(
            columns={'normalized_config': 'config'}
        )
    else:
        grouped_df = conc_res_df.groupby('config').agg({
            'r2': 'std',
            'mse': 'std',
            'mae': 'std',
        }).reset_index()

    grouped_df.to_csv(output_csv_path, index=False)
    return grouped_df