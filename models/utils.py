import os
import re

import pandas as pd
import torch
from torch.utils.data import Subset
from glob import glob

from ..logging_information.logging_config import get_logger
from ..data_processing.dataset_loader import obtain_train_data
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


def obtain_config_dict(config_str: str):
    config = config_str.strip("{}")
    config = [x.split(": ") for x in config.split(", '")]
    config = [(x[0].replace("'", ''), x[1]) for x in config]
    config = dict(config)
    config["model_class"] = str(config["model_class"]).split(
                '.'
            )[-1].replace("'>", "")
    lag_fh = config['lag_forecast']
    lag_fh = lag_fh.replace("'", "")
    lag_forecast = lag_fh.split(",")
    lag_forecast = [value.strip("[] ").strip() for value in lag_forecast]
    config["lag"] = lag_forecast[0]
    config["forecast_horizon"] = lag_forecast[1]
    return config


def normalize_config(config_str: str, remove_key_str: list):
    try:
        config = obtain_config_dict(config_str)
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
            re.search(r'seed(\d+|None)\.csv$', f))
    ])


def average_results(
        results_dir: str,
        output_csv_path: str,
        model_results: str = 'Linear'
):
    file_list = get_seed_file_list(results_dir)

    if not file_list:
        raise ValueError("No CSV files found in directory")

    if model_results == 'Linear':
        df_list = []
        for f in file_list:
            df = pd.read_csv(f)

            df['source_file'] = os.path.basename(f)
            try:
                df['config'] = df['config'].apply(obtain_config_dict)
                config_df = pd.json_normalize(df['config'])
                df = pd.concat(
                    [df.drop('config', axis=1), config_df], axis=1
                )
            except KeyError:
                pass

            df_list.append(df)

        conc_df = pd.concat(df_list, ignore_index=True)

        for col in ['lag', 'forecast_horizon', 'batch_size', 'lr']:
            conc_df[col] = pd.to_numeric(conc_df[col], errors='coerce')

        grouped = conc_df.groupby([
            'model_class',
            'lag',
            'forecast_horizon',
            'batch_size',
            'lr',
        ]).agg(
            r2_mean=('r2', 'mean'),
            r2_std=('r2', 'std'),
            mse_mean=('mse', 'mean'),
            mse_std=('mse', 'std'),
            mae_mean=('mae', 'mean'),
            mae_std=('mae', 'std')
        ).reset_index()
    else:
        conc_df = pd.concat(
            [pd.read_csv(f) for f in file_list],
            ignore_index=True
        )

        grouped = conc_df.groupby('config').agg(
            r2_mean=('r2', 'mean'),
            r2_std=('r2', 'std'),
            mse_mean=('mse', 'mean'),
            mse_std=('mse', 'std'),
            mae_mean=('mae', 'mean'),
            mae_std=('mae', 'std')
        ).reset_index()

    grouped['r2_cv_percent'] = (grouped['r2_std'] / grouped['r2_mean']) * 100
    grouped['mse_cv_percent'] = (grouped['mse_std'] / grouped['mse_mean']) * 100
    grouped['mae_cv_percent'] = (grouped['mae_std'] / grouped['mae_mean']) * 100

    grouped.to_csv(output_csv_path, index=False)
    return grouped


def get_best_results(
        results_dir: str,
        output_csv_path: str,
        model_results: str = 'Linear'
):
    file_list = get_seed_file_list(results_dir)

    if model_results == 'Linear':
        df_list = []
        for f in file_list:
            df = pd.read_csv(f)

            df['source_file'] = os.path.basename(f)
            try:
                df['config'] = df['config'].apply(obtain_config_dict)
                config_df = pd.json_normalize(df['config'])
                df = pd.concat(
                    [df.drop('config', axis=1), config_df], axis=1
                )
            except KeyError:
                pass

            df_list.append(df)

        conc_df = pd.concat(df_list, ignore_index=True)

        for col in ['lag', 'forecast_horizon', 'batch_size', 'lr']:
            conc_df[col] = pd.to_numeric(conc_df[col], errors='coerce')

        best_idx = conc_df.groupby(
            ['model_class', 'lag', 'forecast_horizon']
        )['mse'].idxmin()
    else:
        df_list = []
        for f in file_list:
            df = pd.read_csv(f)
            df['source_file'] = os.path.basename(f)
            df_list.append(df)

        conc_df = pd.concat(df_list, ignore_index=True)
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
        denorm_df = reverse_z_standardization(
            file_df,
            denorm_cols,
            original_mean,
            original_std
        )

        denorm_df["wind_speed"] = denorm_df["wind_speed"].apply(mps_to_knots)

        logger.info(f"Df ({file_name}) after posprocess:\n{denorm_df}")

        os.makedirs(dir_output, exist_ok=True)
        file_output = os.path.join(dir_output, f"{file_name}.csv")
        denorm_df.to_csv(file_output, index=False)


def remove_segment_from_filename(file_name: str):
    return re.sub(r'_segment_\d', '', file_name)


def mps_to_knots(speed_mps):
    return speed_mps * 1.94384


def postprocess_data(
        df: pd.DataFrame,
        dir_original_source: str,
        dir_output: str,
        lag: int,
        fh: int,
        random_seed: int,
        file_suffix: str = None,
        original_norm_cols: list = [
            "u_component", "v_component"
        ],
        train_ratio: float = 0.8
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

    df["file_name"] = df.apply(
        lambda row: remove_segment_from_filename(
            row["file_name"]
        ), axis=1
    )

    agg_dict = {col: "mean" for col in mean_cols}
    agg_dict.update({col: "first" for col in constant_cols})

    grouped_df = df.groupby(group_cols).agg(agg_dict).reset_index()

    # You have to compute the mean and std just with the training set.
    train_data = obtain_train_data(
        dir_source=dir_original_source,
        lag=lag,
        forecast_horizon=fh,
        random_seed=random_seed,
        train_ratio=train_ratio
    )

    original_mean = train_data[original_norm_cols].mean()
    original_std = train_data[original_norm_cols].std()

    train_data.to_csv(f"PostprocessTrainDF_{lag}_{fh}.csv", index=False)

    denorm_cols = {
        "true_u_component": "u_component",
        "pred_u_component": "u_component",
        "true_v_component": "v_component",
        "pred_v_component": "v_component",
    }

    for file_name, file_df in grouped_df.groupby("file_name"):
        x = file_df[["true_u_component", "true_v_component", "pred_u_component", "pred_v_component"]]
        
        denorm_df = reverse_z_standardization(
            file_df,
            denorm_cols,
            original_mean,
            original_std
        )

        denorm_df["original_wind_speed"] = denorm_df["wind_speed"]
        denorm_df["original_wind_direction"] = denorm_df["wind_direction"]

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

        denorm_df["wind_speed"] = denorm_df["wind_speed"].apply(mps_to_knots)
        denorm_df["original_wind_speed"] = denorm_df["original_wind_speed"].apply(mps_to_knots)

        x = denorm_df[["true_u_component", "true_v_component", "pred_u_component", "pred_v_component"]]

        if file_suffix:
            output_filename = file_name.replace(".txt_processed.csv", file_suffix)
        else:
            output_filename = file_name

        os.makedirs(dir_output, exist_ok=True)
        file_output = os.path.join(dir_output, output_filename)
        denorm_df.to_csv(file_output, index=False)


def obtain_pred_vs_trues_best_models(
        best_results_path: str,
        base_results_path: str,
        dir_original_source: str,
        base_dir_output: str,
        original_norm_cols: list = [
            "u_component", "v_component"
        ],
        model: str = 'Linear'
):
    best_results_df = pd.read_csv(best_results_path)

    for idx, row in best_results_df.iterrows():
        if model == 'Linear':
            try:
                config = obtain_config_dict(row["config"])
            except KeyError:
                config = row

            seed = config["random_seed"]
            seed = "None" if pd.isna(seed) else str(int(seed)) if isinstance(seed, float) else str(seed)

            model_name = config["model_class"]
            lag = config["lag"]
            fh = config["forecast_horizon"]

            batch_size = config["batch_size"]
            lr = config["lr"]

            file_path = (
                f"seed{seed}/model_{model_name}/lag{lag}_fh{fh}/"
                f"batch_size{batch_size}_lr{lr}/trues_pred_results.csv"
            )
            file_path = os.path.join(base_results_path, file_path)
            trues_pred_df = pd.read_csv(file_path)

            file_output = (
                f"seed{seed}_model{model_name}_lag{lag}_fh{fh}_"
                f"batch_size{batch_size}_lr{lr}"
            )
            dir_output = os.path.join(base_dir_output, file_output)

        elif model == 'PatchTST':
            seed = row["random_seed"]
            seed = "None" if pd.isna(seed) else str(int(seed)) if isinstance(seed, float) else str(seed)

            model_name = row["model_class"]
            try:
                lag = int(row["lag"]) if row["lag"].is_integer() else row["lag"]
                fh = int(row["forecast_horizon"]) if row["forecast_horizon"].is_integer() else row["forecast_horizon"]
                patch_size = int(row["patch_size"]) if row["patch_size"].is_integer() else row["patch_size"]
            except:
                lag = row["lag"]
                fh = row["forecast_horizon"]
                patch_size = row["patch_size"]
            
            batch_size = row["batch_size"]
            lr = row["lr"]

            num_input_channels = row["num_input_channels"]
            patch_stride = row["patch_stride"]
            num_hidden_layers = row["num_hidden_layers"]
            d_model = row["d_model"]
            num_attention_heads = row["num_attention_heads"]
            share_embedding = row["share_embedding"]
            channel_attention = row["channel_attention"]
            ffn_dim = row["ffn_dim"]
            norm_type = row["norm_type"]
            norm_eps = row["norm_eps"]
            activation_function = row["activation_function"]
            pre_norm = row["pre_norm"]
            num_targets = row["num_targets"]
            attention_dropout = row["attention_dropout"]
            positional_dropout = row["positional_dropout"]
            path_dropout = row["path_dropout"]
            head_dropout = row["head_dropout"]

            file_path = os.path.join(
                f"seed{seed}", f"model_{model_name}"
            )
            file_path = os.path.join(
                file_path,
                f"lag{lag}_fh{fh}"
            )
            file_path = os.path.join(
                file_path, f"patch_size{patch_size}"
            )
            file_path = os.path.join(
                file_path, f"batch_size{int(batch_size)}_lr{lr}"
            )
            file_path = os.path.join(
                file_path,
                f"chs{int(num_input_channels)}_stride{int(patch_stride)}"
                f"_layers{int(num_hidden_layers)}_dm{int(d_model)}_heads{int(num_attention_heads)}"
                f"_shareEmb{int(share_embedding)}_chanAtt{int(channel_attention)}_ffn{int(ffn_dim)}"
                f"_norm{norm_type}_eps{norm_eps}_act{activation_function}_preNorm{int(pre_norm)}"
                f"_tgt{int(num_targets)}_attnDO{attention_dropout:.1f}_posDO{positional_dropout:.1f}"
                f"_pathDO{path_dropout:.1f}_headDO{head_dropout:.1f}"
            )
            file_trues_pred = os.path.join(base_results_path, f"{file_path}/trues_pred_results.csv")
            trues_pred_df = pd.read_csv(file_trues_pred)

            dir_output = os.path.join(base_dir_output, file_path)

        elif model == 'Transformer':
            seed = row["random_seed"]
            seed = "None" if pd.isna(seed) else str(int(seed)) if isinstance(seed, float) else str(seed)

            model_name = row["model_class"]
            try:
                lag = int(row["lag"]) if row["lag"].is_integer() else row["lag"]
                fh = int(row["forecast_horizon"]) if row["forecast_horizon"].is_integer() else row["forecast_horizon"]
            except:
                lag = row["lag"]
                fh = row["forecast_horizon"]

            batch_size = row["batch_size"]
            lr = row["lr"]

            num_input_channels = row["num_input_channels"]
            num_hidden_layers = row["num_hidden_layers"]
            d_model = row["d_model"]
            num_attention_heads = row["num_attention_heads"]
            ffn_dim = row["ffn_dim"]
            activation_function = row["activation_function"]
            num_targets = row["num_targets"]
            attention_dropout = row["attention_dropout"]
            positional_dropout = row["positional_dropout"]

            file_path = os.path.join(
                f"seed{seed}", f"model_{model_name}"
            )
            file_path = os.path.join(
                file_path,
                f"lag{lag}_fh{fh}"
            )
            file_path = os.path.join(
                file_path, f"batch_size{int(batch_size)}_lr{lr}"
            )
            file_path = os.path.join(
                file_path,
                f"chs{int(num_input_channels)}_layers{int(num_hidden_layers)}_dm{int(d_model)}"
                f"_heads{int(num_attention_heads)}_ffn{int(ffn_dim)}"
                f"_act{activation_function}_tgt{int(num_targets)}"
                f"_attnDO{attention_dropout:.1f}_posDO{positional_dropout:.1f}"
            )
            file_trues_pred = os.path.join(base_results_path, f"{file_path}/trues_pred_results.csv")
            trues_pred_df = pd.read_csv(file_trues_pred)

            dir_output = os.path.join(base_dir_output, file_path)

        elif model == 'VAR':
            model_name = 'VAR'
            config = row["config"]
            lag_forecast = config.split(",")
            lag_forecast = [
                value.strip(
                    "() "
                ).strip() for value in lag_forecast
            ]
            lag = int(lag_forecast[0])
            fh = int(lag_forecast[1])
            source_file = row["source_file"]
            seed = (source_file.split("_"))[2]
            seed = (seed.split("."))[0]

            seed = seed.split("seed")[-1]  # Gets "None.csv" or "0.csv"
            seed = seed.split(".")[0]    # Gets "None" or "0"
            
            # Convert "None" to Python None, otherwise return as int
            if seed == "None":
                seed = None
            else:
                seed = int(seed)

            file_path = (
                f"seed{seed}/model_VAR/lag{lag}_fh{fh}/"
                "trues_pred_results.csv"
            )

            file_path = os.path.join(base_results_path, file_path)
            trues_pred_df = pd.read_csv(file_path)

            file_output = (f"seed{seed}_model_VAR_lag{lag}_fh{fh}")
            dir_output = os.path.join(base_dir_output, file_output)
        
        file_suffix = f"_{model_name}_l{lag}_h{fh}.csv"
        postprocess_data(
            df=trues_pred_df,
            dir_original_source=dir_original_source,
            dir_output=dir_output,
            file_suffix=file_suffix,
            original_norm_cols=original_norm_cols,
            lag=lag,
            fh=fh,
            random_seed=seed
        )


def concatenate_all_seeds_results(results_dir: str, output_csv_path: str = None):
    file_list = get_seed_file_list(results_dir)

    if not file_list:
        raise ValueError("No CSV files found in directory")

    df_list = []
    for f in file_list:
        df = pd.read_csv(f)

        df['source_file'] = os.path.basename(f)
        try:
            df['config'] = df['config'].apply(obtain_config_dict)
            config_df = pd.json_normalize(df['config'])
            df = pd.concat(
                [df.drop('config', axis=1), config_df], axis=1
            )
        except KeyError:
            pass

        df_list.append(df)

    conc_df = pd.concat(df_list, ignore_index=True)

    if output_csv_path is not None:
        conc_df.to_csv(output_csv_path, index=False)

    return conc_df


def get_num_timesteps(dir_source: str, output_file: str):
    file_list = glob(os.path.join(dir_source, "*.csv"))
    timesteps_info = {}

    for file_path in file_list:
        df = pd.read_csv(
            file_path, parse_dates=["timestamp"],
            dtype={"file_name": str}, low_memory=False
        )
        timesteps_info[os.path.basename(file_path)] = len(df) - 1

    summary_df = pd.DataFrame(list(timesteps_info.items()), columns=["file_name", "num_timesteps"])
    summary_df.to_csv(os.path.join(dir_source, output_file), index=False)
