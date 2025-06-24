import os

import matplotlib.pyplot as plt
import pandas as pd
from typing import List

from .utils import obtain_config_dict


def get_config(row):
    config = row["config"]
    config = config.strip("{}")
    config = [x.split(": ") for x in config.split(", '")]
    config = [(x[0].replace("'", ''), x[1]) for x in config]
    config = dict(config)
    return config


def get_lag(row):
    config = get_config(row)
    lag_fh = config['lag_forecast']
    lag_fh = lag_fh.replace("'", "")
    lag_forecast = lag_fh.split(",")
    lag_forecast = [value.strip("[] ").strip() for value in lag_forecast]
    return lag_forecast[0]


def get_forecast_horizon(row):
    config = get_config(row)
    lag_fh = config['lag_forecast']
    lag_fh = lag_fh.replace("'", "")
    lag_forecast = lag_fh.split(",")
    lag_forecast = [value.strip("[] ").strip() for value in lag_forecast]
    return lag_forecast[1]


def get_model_class(row):
    config = get_config(row)
    model_class = str(config["model_class"]).split(
        '.'
    )[-1].replace("'>", "")
    return model_class


def get_batch_size(row):
    config = get_config(row)
    return config["batch_size"]


def get_learning_rate(row):
    config = get_config(row)
    return config["lr"]


def get_best_patchtst_results(dataframe: pd.DataFrame, min_by: str = 'mse_mean'):
    try:
        dataframe["lag"] = dataframe.apply(get_lag, axis=1)
        dataframe["forecast_horizon"] = dataframe.apply(get_forecast_horizon, axis=1)
        dataframe["model_class"] = dataframe.apply(get_model_class, axis=1)
        dataframe["batch_size"] = dataframe.apply(get_batch_size, axis=1)
        dataframe["lr"] = dataframe.apply(get_learning_rate, axis=1)
        dataframe["patch_size"] = dataframe.apply(lambda x: x.get("patch_size", None), axis=1)
    except KeyError:
        pass

    grouped_df = dataframe.loc[
        dataframe.groupby(
            ['lag', 'forecast_horizon', 'model_class']
        )[min_by].idxmin()
    ].reset_index()

    results_info = {}
    for (lag, forecast_horizon, model_class, mse, mae, r2,
         batch_size, lr, patch_size, num_input_channels, patch_stride,
         num_hidden_layers, d_model, num_attention_heads, share_embedding,
         channel_attention, ffn_dim, norm_type, norm_eps, activation_function,
         pre_norm, num_targets, attention_dropout, positional_dropout,
         path_dropout, head_dropout) in zip(
        grouped_df['lag'], grouped_df['forecast_horizon'],
        grouped_df['model_class'], grouped_df['mse_mean'],
        grouped_df['mae_mean'], grouped_df['r2_mean'],
        grouped_df['batch_size'], grouped_df['lr'],
        grouped_df['patch_size'], grouped_df['num_input_channels'],
        grouped_df['patch_stride'], grouped_df['num_hidden_layers'],
        grouped_df['d_model'], grouped_df['num_attention_heads'],
        grouped_df['share_embedding'], grouped_df['channel_attention'],
        grouped_df['ffn_dim'], grouped_df['norm_type'],
        grouped_df['norm_eps'], grouped_df['activation_function'],
        grouped_df['pre_norm'], grouped_df['num_targets'],
        grouped_df['attention_dropout'], grouped_df['positional_dropout'],
        grouped_df['path_dropout'], grouped_df['head_dropout']
    ):
        results_info.setdefault(forecast_horizon, {})
        results_info[forecast_horizon].setdefault(lag, {})
        results_info[forecast_horizon][lag].setdefault(model_class, {})
        results_info[forecast_horizon][lag][model_class] = {
            "r2_mean": r2,
            "mae_mean": mae,
            "mse_mean": mse,
            "batch_size": batch_size,
            "lr": lr,
            "patch_size": patch_size,
            "num_input_channels": num_input_channels,
            "patch_stride": patch_stride,
            "num_hidden_layers": num_hidden_layers,
            "d_model": d_model,
            "num_attention_heads": num_attention_heads,
            "share_embedding": share_embedding,
            "channel_attention": channel_attention,
            "ffn_dim": ffn_dim,
            "norm_type": norm_type,
            "norm_eps": norm_eps,
            "activation_function": activation_function,
            "pre_norm": pre_norm,
            "num_targets": num_targets,
            "attention_dropout": attention_dropout,
            "positional_dropout": positional_dropout,
            "path_dropout": path_dropout,
            "head_dropout": head_dropout
        }
    return results_info


def get_best_transformer_results(dataframe: pd.DataFrame, min_by: str = 'mse_mean'):
    try:
        dataframe["lag"] = dataframe.apply(get_lag, axis=1)
        dataframe["forecast_horizon"] = dataframe.apply(get_forecast_horizon, axis=1)
        dataframe["model_class"] = dataframe.apply(get_model_class, axis=1)
        dataframe["batch_size"] = dataframe.apply(get_batch_size, axis=1)
        dataframe["lr"] = dataframe.apply(get_learning_rate, axis=1)
    except KeyError:
        pass

    grouped_df = dataframe.loc[
        dataframe.groupby(
            ['lag', 'forecast_horizon', 'model_class']
        )[min_by].idxmin()
    ].reset_index()

    results_info = {}
    for (lag, forecast_horizon, model_class, mse, mae, r2,
         batch_size, lr, num_input_channels,
         num_hidden_layers, d_model, num_attention_heads, 
         ffn_dim, activation_function,
         num_targets, attention_dropout, positional_dropout,) in zip(
        grouped_df['lag'], grouped_df['forecast_horizon'],
        grouped_df['model_class'], grouped_df['mse_mean'],
        grouped_df['mae_mean'], grouped_df['r2_mean'],
        grouped_df['batch_size'], grouped_df['lr'],
        grouped_df['num_input_channels'], grouped_df['num_hidden_layers'],
        grouped_df['d_model'], grouped_df['num_attention_heads'],
        grouped_df['ffn_dim'], grouped_df['activation_function'],
        grouped_df['num_targets'], grouped_df['attention_dropout'], 
        grouped_df['positional_dropout']
    ):
        results_info.setdefault(forecast_horizon, {})
        results_info[forecast_horizon].setdefault(lag, {})
        results_info[forecast_horizon][lag].setdefault(model_class, {})
        results_info[forecast_horizon][lag][model_class] = {
            "r2_mean": r2,
            "mae_mean": mae,
            "mse_mean": mse,
            "batch_size": batch_size,
            "lr": lr,
            "num_input_channels": num_input_channels,
            "num_hidden_layers": num_hidden_layers,
            "d_model": d_model,
            "num_attention_heads": num_attention_heads,
            "ffn_dim": ffn_dim,
            "activation_function": activation_function,
            "num_targets": num_targets,
            "attention_dropout": attention_dropout,
            "positional_dropout": positional_dropout,
        }
    return results_info


def get_best_linear_results(dataframe: pd.DataFrame, min_by: str = 'mse_mean'):
    try:
        dataframe["lag"] = dataframe.apply(get_lag, axis=1)
        dataframe["forecast_horizon"] = dataframe.apply(get_forecast_horizon, axis=1)
        dataframe["model_class"] = dataframe.apply(get_model_class, axis=1)
        dataframe["batch_size"] = dataframe.apply(get_batch_size, axis=1)
        dataframe["lr"] = dataframe.apply(get_learning_rate, axis=1)
    except KeyError:
        pass

    grouped_df = dataframe.loc[
        dataframe.groupby(
            ['lag', 'forecast_horizon', 'model_class']
        )[min_by].idxmin()
    ].reset_index()

    results_info = {}
    for lag, forecast_horizon, model_class, mse, mae, r2, batch_size, lr in zip(
        grouped_df['lag'], grouped_df['forecast_horizon'],
        grouped_df['model_class'], grouped_df['mse_mean'],
        grouped_df['mae_mean'], grouped_df['r2_mean'],
        grouped_df['batch_size'], grouped_df['lr']
    ):
        results_info.setdefault(forecast_horizon, {})
        results_info[forecast_horizon].setdefault(lag, {})
        results_info[forecast_horizon][lag].setdefault(model_class, {})
        results_info[forecast_horizon][lag][model_class] = {
            "r2_mean": r2,
            "mae_mean": mae,
            "mse_mean": mse,
            "batch_size": batch_size,
            "lr": lr
        }
    return results_info


def plot_best_patchtst_results(
    dataframe: pd.DataFrame,
    base_dir_out: str,
    metrics_to_include: List[str] = ["mse_mean"],
    show_plot: bool = False
) -> None:
    results_info = get_best_patchtst_results(dataframe)
    os.makedirs(base_dir_out, exist_ok=True)

    # Define plot styles for different metrics
    metric_styles = {
        "r2_mean": {"linestyle": ":", "marker": "o", "suffix": " - R2"},
        "mae_mean": {"linestyle": "--", "marker": "o", "suffix": " - MAE"},
        "mse_mean": {"linestyle": "-", "marker": "o", "suffix": "- MSE"}
    }

    for forecast_horizon, lag_data in results_info.items():
        forecast_horizon = int(forecast_horizon)
        ordered_lag_data = dict(sorted(lag_data.items(), key=lambda item: int(item[0])))
        lags = [int(lag) for lag in ordered_lag_data.keys()]
        model_classes = list(next(iter(ordered_lag_data.values())).keys())

        plt.figure(figsize=(16, 10))

        for model_class in sorted(model_classes):
            for metric in metrics_to_include:
                if metric not in metric_styles:
                    continue
                    
                # Get values for the current metric
                values = [
                    ordered_lag_data[lag][model_class][metric] 
                    for lag in lags
                ]
                
                # Generate configuration labels
                config_labels = []
                for lag in lags:
                    config = ordered_lag_data[lag][model_class]
                    label = (
                        f"Lag {lag}: "
                        f"Patch: {int(config['patch_size'])}, "
                        f"Stride: {int(config['patch_stride'])}, \n"
                        f"Layers: {int(config['num_hidden_layers'])}, "
                        f"d_model: {int(config['d_model'])}\n"
                        f"Heads: {int(config['num_attention_heads'])}, "
                        f"FFN: {int(config['ffn_dim'])}, \n"
                        f"Batch: {int(config['batch_size'])}, "
                        f"LR: {config['lr']:.0e}, \n"
                        f"Norm eps: {config['norm_eps']*100}, "
                        f"Pos Dropout: {config['positional_dropout']*100}, \n"
                        f"Head Dropout: {config['head_dropout']*100}, "
                        f"Path Dropout: {config['path_dropout']*100}, \n"
                        f"Att Dropout: {config['attention_dropout']*100}"
                        f"{metric_styles[metric]['suffix']}"
                    )
                    config_labels.append(label)
                
                # Plot the values
                plt.plot(
                    lags, values,
                    label="\n".join(config_labels),
                    linestyle=metric_styles[metric]["linestyle"],
                    marker=metric_styles[metric]["marker"]
                )

        plt.title(f"Forecast Horizon: {forecast_horizon}")
        plt.xlabel("Lag")
        plt.ylabel("Metric Value")
        
        # Move legend outside the plot to the right
        plt.legend(
            title="Patch Size Configurations",
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.,
            fontsize='small'
        )
        
        plt.grid(True)
        plt.tight_layout(rect=[0, 0, 0.75, 1])  # Left, bottom, right, top

        plt.savefig(
            f"{base_dir_out}/fh{forecast_horizon}_results.png",
            bbox_inches='tight'
        )
        
        if show_plot:
            plt.show()
        else:
            plt.close()


def plot_best_transformer_results(
    dataframe: pd.DataFrame,
    base_dir_out: str,
    metrics_to_include: List[str] = ["mse_mean"],
    show_plot: bool = False
) -> None:
    results_info = get_best_transformer_results(dataframe)
    os.makedirs(base_dir_out, exist_ok=True)

    METRIC_CONFIG = {
        "r2_mean": {"style": ":", "suffix": " - R2"},
        "mae_mean": {"style": "--", "suffix": " - MAE"},
        "mse_mean": {"style": "-", "suffix": " - MSE"}
    }

    for forecast_horizon, lag_data in results_info.items():
        forecast_horizon = int(forecast_horizon)
        ordered_lag_data = dict(sorted(lag_data.items(), key=lambda item: int(item[0])))
        lags = [int(lag) for lag in ordered_lag_data.keys()]
        model_classes = list(next(iter(ordered_lag_data.values())).keys())

        plt.figure(figsize=(16, 10))

        for model_class in sorted(model_classes):
            for metric in metrics_to_include:
                if metric not in METRIC_CONFIG:
                    continue
                
                # Get values and prepare labels
                values = [ordered_lag_data[lag][model_class][metric] for lag in lags]
                config_labels = []
                
                for lag in lags:
                    config = ordered_lag_data[lag][model_class]
                    label = (
                        f"Lag {lag}: "
                        f"Layers: {int(config['num_hidden_layers'])}, "
                        f"d_model: {int(config['d_model'])}\n"
                        f"Heads: {int(config['num_attention_heads'])}, "
                        f"FFN: {int(config['ffn_dim'])}, \n"
                        f"Batch: {int(config['batch_size'])}, "
                        f"LR: {config['lr']:.0e}, \n"
                        f"Pos Dropout: {config['positional_dropout']}, \n"
                        f"Att Dropout: {config['attention_dropout']}"
                        f"{METRIC_CONFIG[metric]['suffix']}"
                    )
                    config_labels.append(label)
                
                # Plot the data
                plt.plot(
                    lags, values,
                    label="\n".join(config_labels),
                    linestyle=METRIC_CONFIG[metric]["style"],
                    marker='o'
                )

        plt.title(f"Forecast Horizon: {forecast_horizon}")
        plt.xlabel("Lag")
        plt.ylabel("Metric Value")
        plt.grid(True)
        
        plt.legend(
            title="Model Configurations",
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.,
            fontsize='small'
        )
        
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        plt.savefig(
            f"{base_dir_out}/fh{forecast_horizon}_results.png",
            bbox_inches='tight'
        )
        
        if show_plot:
            plt.show()
        plt.close()


def plot_all_patchtst_results_joined(
        dataframe: pd.DataFrame,
        base_dir_out: str,
        metric: str = "mse_mean",
        show_plot: bool = False,
):
    if not os.path.exists(base_dir_out):
        os.makedirs(base_dir_out)

    config_cols = [
        'patch_size', 'num_hidden_layers', 'd_model', 'num_attention_heads',
        'ffn_dim', 'batch_size', 'lr', 'attention_dropout'
    ]

    forecast_horizons = sorted(dataframe['forecast_horizon'].unique())

    for fh in forecast_horizons:
        fh_data = dataframe[dataframe['forecast_horizon'] == fh]

        fig, ax = plt.subplots(figsize=(20, 12))

        grouped = fh_data.groupby(config_cols)
        handles = []
        labels = []

        for config, group in grouped:
            group = group.sort_values("lag")
            if metric not in group.columns:
                continue

            config_label = (
                f"Patch: {config[0]}, "
                f"Layers: {config[1]}, d_model: {config[2]}, "
                f"Heads: {config[3]}, FFN: {config[4]}, "
                f"Batch: {config[5]}, LR: {config[6]}, "
                f"Att Dropout: {config[7]}"
            )

            line, = ax.plot(
                group['lag'],
                group[metric],
                marker='o',
                linewidth=2,
                label=config_label
            )

            handles.append(line)
            labels.append(config_label)

        ax.set_title(f"Performance Across Lags — Forecast Horizon: {fh}", fontsize=16)
        ax.set_xlabel("Lag", fontsize=14)
        ax.set_ylabel(metric, fontsize=14)
        ax.grid(True)

        # Save main plot without legend
        plot_path = os.path.join(base_dir_out, f"fh{fh}_config_lines.png")
        fig.tight_layout()
        fig.savefig(plot_path, bbox_inches='tight')
        if show_plot:
            plt.show()
        plt.close(fig)

        # Save legend separately
        legend_fig = plt.figure(figsize=(12, len(labels) * 0.3))
        legend_fig.canvas.draw()

        legend_path = os.path.join(base_dir_out, f"fh{fh}_legend.png")
        legend_fig.savefig(legend_path, bbox_inches='tight', dpi=300)
        plt.close(legend_fig)


def plot_best_patchtst_results_by_patch_size(
        dataframe: pd.DataFrame,
        base_dir_out: str,
        metrics_to_include: list = ["mse_mean"],
        show_plot: bool = False
):   
    all_patch_sizes = sorted(dataframe['patch_size'].unique())
    
    grouped = dataframe.groupby(
        ['forecast_horizon', 'lag', 'patch_size']
    )
    
    best_by_patch = grouped.apply(
        lambda x: x.loc[x['mse_mean'].idxmin()]
    ).reset_index(drop=True)
    
    if not os.path.exists(base_dir_out):
        os.makedirs(base_dir_out)
    
    for forecast_horizon in best_by_patch['forecast_horizon'].unique():
        forecast_horizon = int(forecast_horizon)
        plt.figure(figsize=(16, 10))
        
        fh_data = best_by_patch[best_by_patch['forecast_horizon'] == forecast_horizon]
        
        lags = sorted(fh_data['lag'].unique())
        
        for patch_size in all_patch_sizes:
            patch_data = fh_data[fh_data['patch_size'] == patch_size]
            
            if patch_data.empty:
                continue

            patch_data = patch_data.sort_values('lag')
            
            for metric in metrics_to_include:
                if metric not in patch_data.columns:
                    continue
                
                config_label = f"Patch: {patch_size}\n"
                
                for x, row in patch_data.iterrows():
                    if metric == "mse_mean":
                        metric_label = "MSE"
                    elif metric == "mae_mean":
                        metric_label = "MAE"
                    elif metric == "r2_mean":
                        metric_label = "R2"

                    config_label = (
                        f"{config_label}"
                        f"Lag {int(row['lag'])}: "
                        f"Layers: {int(row['num_hidden_layers'])}, "
                        f"Stride: {int(row['patch_stride'])}, "
                        f"d_model: {int(row['d_model'])}\n"
                        f"Heads: {int(row['num_attention_heads'])}, "
                        f"FFN: {int(row['ffn_dim'])}\n"
                        f"Batch: {int(row['batch_size'])}, "
                        f"LR: {int(row['lr'])}\n"
                        f"Att Dropout: {row['attention_dropout']} - {metric_label}\n"
                    )
                
                plt.plot(
                    patch_data['lag'],
                    patch_data[metric],
                    label=config_label,
                    marker='o',
                    linewidth=2
                )
        
        plt.title(f"Best Performance by Patch Size - Forecast Horizon: {int(forecast_horizon)}")
        plt.xlabel("Lag")
        plt.ylabel("Metric Value")
        plt.xticks(lags)
        plt.grid(True)
        
        # Move legend outside the plot to the right
        plt.legend(
            title="Patch Size Configurations",
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.,
            fontsize='small'
        )
        
        # Adjust layout to make room for the legend
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        
        # Save the plot
        plt.savefig(
            f"{base_dir_out}/fh{int(forecast_horizon)}_by_patch_size.png",
            bbox_inches='tight'
        )
        
        if show_plot:
            plt.show()
        plt.close()


def plot_patchtst_comparison_results(
        uncapped_df: pd.DataFrame,
        capped_df: pd.DataFrame,
        base_dir_out: str,
        metrics_to_include=["mse_mean"],
        show_plot=False
):
    uncapped_results = get_best_patchtst_results(uncapped_df)
    capped_results = get_best_patchtst_results(capped_df)

    for forecast_horizon in uncapped_results.keys():

        fh_int = int(forecast_horizon)
        if fh_int not in capped_results:
            continue

        plt.figure(figsize=(14, 8))

        uncapped_lag_data = uncapped_results[forecast_horizon]
        ordered_uncapped = {int(k): v for k, v in sorted(uncapped_lag_data.items(), key=lambda item: int(item[0]))}
        uncapped_lags = list(ordered_uncapped.keys())

        capped_lag_data = capped_results[fh_int]
        ordered_capped = {int(k): v for k, v in sorted(capped_lag_data.items(), key=lambda item: int(item[0]))}
        capped_lags = list(ordered_capped.keys())

        common_lags = sorted(list(set(uncapped_lags) & set(capped_lags)))
        if not common_lags:
            plt.close()
            continue

        model_classes = list(next(iter(ordered_uncapped.values())).keys())

        for model_class in sorted(model_classes):
            if "r2_mean" in metrics_to_include:
                r2_values = [ordered_uncapped[lag][model_class]["r2_mean"] for lag in common_lags]
                plt.plot(common_lags, r2_values, label=f"Uncapped {model_class} - R2", linestyle=":", marker='o')

            if "mae_mean" in metrics_to_include:
                mae_values = [ordered_uncapped[lag][model_class]["mae_mean"] for lag in common_lags]
                plt.plot(common_lags, mae_values, label=f"Uncapped {model_class} - MAE", linestyle="--", marker='o')

            if "mse_mean" in metrics_to_include:
                mse_values = [ordered_uncapped[lag][model_class]["mse_mean"] for lag in common_lags]
                plt.plot(common_lags, mse_values, label=f"Uncapped {model_class}", linestyle="-", marker='o')

        for model_class in sorted(model_classes):
            if "r2_mean" in metrics_to_include:
                r2_values = [ordered_capped[lag][model_class]["r2_mean"] for lag in common_lags]
                plt.plot(common_lags, r2_values, label=f"Capped {model_class} - R2", linestyle=":", marker='x')

            if "mae_mean" in metrics_to_include:
                mae_values = [ordered_capped[lag][model_class]["mae_mean"] for lag in common_lags]
                plt.plot(common_lags, mae_values, label=f"Capped {model_class} - MAE", linestyle="--", marker='x')

            if "mse_mean" in metrics_to_include:
                mse_values = [ordered_capped[lag][model_class]["mse_mean"] for lag in common_lags]
                plt.plot(common_lags, mse_values, label=f"Capped {model_class}", linestyle="-", marker='x')

        plt.title(f"Forecast Horizon: {forecast_horizon} - Capped vs Uncapped Comparison")
        plt.xlabel("Lag")
        plt.ylabel("MSE")
        plt.legend(title="Data Type, Model Class and Metric")
        plt.grid(True)
        plt.tight_layout()

        if not os.path.exists(base_dir_out):
            os.makedirs(base_dir_out)

        plt.savefig(f"{base_dir_out}/comparison_fh{forecast_horizon}_results.png")
        if show_plot:
            plt.show()

        plt.close()


def plot_transformer_comparison_results(
        uncapped_df: pd.DataFrame,
        capped_df: pd.DataFrame,
        base_dir_out: str,
        metrics_to_include=["mse_mean"],
        show_plot=False
):
    uncapped_results = get_best_transformer_results(uncapped_df)
    capped_results = get_best_transformer_results(capped_df)

    for forecast_horizon in uncapped_results.keys():

        fh_int = int(forecast_horizon)
        if fh_int not in capped_results:
            continue

        plt.figure(figsize=(14, 8))

        uncapped_lag_data = uncapped_results[forecast_horizon]
        ordered_uncapped = {int(k): v for k, v in sorted(uncapped_lag_data.items(), key=lambda item: int(item[0]))}
        uncapped_lags = list(ordered_uncapped.keys())

        capped_lag_data = capped_results[fh_int]
        ordered_capped = {int(k): v for k, v in sorted(capped_lag_data.items(), key=lambda item: int(item[0]))}
        capped_lags = list(ordered_capped.keys())

        common_lags = sorted(list(set(uncapped_lags) & set(capped_lags)))
        if not common_lags:
            plt.close()
            continue

        model_classes = list(next(iter(ordered_uncapped.values())).keys())

        for model_class in sorted(model_classes):
            if "r2_mean" in metrics_to_include:
                r2_values = [ordered_uncapped[lag][model_class]["r2_mean"] for lag in common_lags]
                plt.plot(common_lags, r2_values, label=f"Uncapped {model_class} - R2", linestyle=":", marker='o')

            if "mae_mean" in metrics_to_include:
                mae_values = [ordered_uncapped[lag][model_class]["mae_mean"] for lag in common_lags]
                plt.plot(common_lags, mae_values, label=f"Uncapped {model_class} - MAE", linestyle="--", marker='o')

            if "mse_mean" in metrics_to_include:
                mse_values = [ordered_uncapped[lag][model_class]["mse_mean"] for lag in common_lags]
                plt.plot(common_lags, mse_values, label=f"Uncapped {model_class}", linestyle="-", marker='o')

        for model_class in sorted(model_classes):
            if "r2_mean" in metrics_to_include:
                r2_values = [ordered_capped[lag][model_class]["r2_mean"] for lag in common_lags]
                plt.plot(common_lags, r2_values, label=f"Capped {model_class} - R2", linestyle=":", marker='x')

            if "mae_mean" in metrics_to_include:
                mae_values = [ordered_capped[lag][model_class]["mae_mean"] for lag in common_lags]
                plt.plot(common_lags, mae_values, label=f"Capped {model_class} - MAE", linestyle="--", marker='x')

            if "mse_mean" in metrics_to_include:
                mse_values = [ordered_capped[lag][model_class]["mse_mean"] for lag in common_lags]
                plt.plot(common_lags, mse_values, label=f"Capped {model_class}", linestyle="-", marker='x')

        plt.title(f"Forecast Horizon: {forecast_horizon} - Capped vs Uncapped Comparison")
        plt.xlabel("Lag")
        plt.ylabel("MSE")
        plt.legend(title="Data Type, Model Class and Metric")
        plt.grid(True)
        plt.tight_layout()

        if not os.path.exists(base_dir_out):
            os.makedirs(base_dir_out)

        plt.savefig(f"{base_dir_out}/comparison_fh{forecast_horizon}_results.png")
        if show_plot:
            plt.show()

        plt.close()



def plot_best_linear_results(
        dataframe: pd.DataFrame,
        base_dir_out: str,
        metrics_to_include: list = ["mse_mean"],
        show_plot: bool = False
):
    results_info = get_best_linear_results(dataframe)

    for forecast_horizon, lag_data in results_info.items():
        ordered_lag_data = dict(sorted(lag_data.items(), key=lambda item: int(item[0])))
        lags = list(ordered_lag_data.keys())

        model_classes = list(next(iter(ordered_lag_data.values())).keys())

        plt.figure(figsize=(14, 8))

        for model_class in sorted(model_classes):
            if "r2_mean" in metrics_to_include:
                r2_values = [
                    ordered_lag_data[lag][model_class]["r2_mean"] for lag in lags
                ]
                config_labels = [
                    f"{model_class} (Batch: {ordered_lag_data[lag][model_class]['batch_size']}, "
                    f"LR: {ordered_lag_data[lag][model_class]['lr']}) - R2"
                    for lag in lags
                ]
                plt.plot(
                    lags, r2_values, label=", ".join(config_labels),
                    linestyle=":", marker='o'
                )
            if "mae_mean" in metrics_to_include:
                mae_values = [
                    ordered_lag_data[lag][model_class]["mae_mean"] for lag in lags
                ]
                config_labels = [
                    f"{model_class} (Batch: {ordered_lag_data[lag][model_class]['batch_size']}, "
                    f"LR: {ordered_lag_data[lag][model_class]['lr']}) - MAE"
                    for lag in lags
                ]
                plt.plot(
                    lags, mae_values, label=", ".join(config_labels),
                    linestyle="--", marker='o'
                )
            if "mse_mean" in metrics_to_include:
                mse_values = [
                    ordered_lag_data[lag][model_class]["mse_mean"] for lag in lags
                ]
                config_labels = [
                    f"{model_class} (Batch: {ordered_lag_data[lag][model_class]['batch_size']}, "
                    f"LR: {ordered_lag_data[lag][model_class]['lr']})"
                    for lag in lags
                ]
                plt.plot(
                    lags, mse_values, label=", ".join(config_labels),
                    linestyle="-", marker='o'
                )

        plt.title(f"Forecast Horizon: {forecast_horizon}")
        plt.xlabel("Lag")
        plt.ylabel("MSE")
        plt.legend(title="Model Class and Metric")
        plt.grid(True)
        plt.tight_layout()

        if not os.path.exists(base_dir_out):
            os.makedirs(base_dir_out)

        plt.savefig(f"{base_dir_out}/fh{forecast_horizon}_results.png")
        if show_plot:
            plt.show()

        plt.clf()


def get_all_linear_results_separated(dataframe: pd.DataFrame):
    results_info = {}

    try:
        dataframe['config'] = dataframe['config'].apply(obtain_config_dict)
        config_df = pd.json_normalize(dataframe['config'])
        dataframe = pd.concat(
            [dataframe.drop('config', axis=1), config_df], axis=1
        )
    except KeyError:
        pass

    for (
        forecast_horizon, batch_size, lr,
        lag, model_class, r2, mae, mse
    ) in zip(
        dataframe['forecast_horizon'], dataframe['batch_size'],
        dataframe['lr'], dataframe['lag'],
        dataframe['model_class'], dataframe['r2_mean'],
        dataframe['mae_mean'], dataframe['mse_mean']
    ):
        results_info.setdefault(batch_size, {})
        results_info[batch_size].setdefault(lr, {})
        results_info[batch_size][lr].setdefault(forecast_horizon, {})
        results_info[batch_size][lr][forecast_horizon].setdefault(lag, {})
        results_info[batch_size][lr][forecast_horizon][lag][model_class] = {
            "r2_mean": r2,
            "mae_mean": mae,
            "mse_mean": mse
        }

    return results_info


def plot_all_linear_results_separated(
        dataframe: pd.DataFrame,
        base_dir_out: str,
        metrics_to_include: list = ["mse_mean"],
        show_plot: bool = False
):
    results_info = get_all_linear_results_separated(dataframe)

    for batch_size, lr_data in results_info.items():
        for lr, forecast_data in lr_data.items():
            for forecast_horizon, lag_data in forecast_data.items():
                ordered_lag_data = dict(sorted(lag_data.items(), key=lambda item: int(item[0])))
                lags = list(ordered_lag_data.keys())

                model_classes = list(next(iter(ordered_lag_data.values())).keys())

                plt.figure(figsize=(14, 8))

                for model_class in sorted(model_classes):
                    if "r2_mean" in metrics_to_include:
                        r2_values = [ordered_lag_data[lag][model_class]["r2_mean"] for lag in lags]
                        plt.plot(lags, r2_values, label=f"{model_class} - R2", linestyle=":", marker = 'o')

                    if "mae_mean" in metrics_to_include:
                        mae_values = [ordered_lag_data[lag][model_class]["mae_mean"] for lag in lags]
                        plt.plot(lags, mae_values, label=f"{model_class} - MAE", linestyle="--", marker = 'o')

                    if "mse_mean" in metrics_to_include:
                        mse_values = [ordered_lag_data[lag][model_class]["mse_mean"] for lag in lags]
                        plt.plot(lags, mse_values, label=f"{model_class}", linestyle="-", marker = 'o')

                plt.title(f"Batch Size: {batch_size}, LR: {lr}, Forecast Horizon: {forecast_horizon}")
                plt.xlabel("Lag")
                plt.ylabel("MSE")
                plt.legend(title="Model Class and Metric")
                plt.grid(True)
                plt.tight_layout()

                specific_dir = f"batch_size{batch_size}/lr{lr}"
                out_path = os.path.join(base_dir_out, specific_dir)

                if not os.path.exists(out_path):
                    os.makedirs(out_path)

                plt.savefig(f"{out_path}/fh{forecast_horizon}_results.png")
                if show_plot:
                    plt.show()

                plt.clf()


def get_all_linear_results_joined(dataframe: pd.DataFrame):
    results_info = {}

    try:
        dataframe['config'] = dataframe['config'].apply(obtain_config_dict)
        config_df = pd.json_normalize(dataframe['config'])
        dataframe = pd.concat(
            [dataframe.drop('config', axis=1), config_df], axis=1
        )
    except KeyError:
        pass

    for (
        forecast_horizon, batch_size, lr,
        lag, model_class, r2, mae, mse
    ) in zip(
        dataframe['forecast_horizon'], dataframe['batch_size'],
        dataframe['lr'], dataframe['lag'],
        dataframe['model_class'], dataframe['r2_mean'],
        dataframe['mae_mean'], dataframe['mse_mean']
    ):
        results_info.setdefault(forecast_horizon, {})
        results_info[forecast_horizon].setdefault(batch_size, {})
        results_info[forecast_horizon][batch_size].setdefault(lr, {})
        results_info[forecast_horizon][batch_size][lr].setdefault(lag, {})
        results_info[forecast_horizon][batch_size][lr][lag][model_class] = {
            "r2_mean": r2,
            "mae_mean": mae,
            "mse_mean": mse
        }

    return results_info


def plot_all_linear_results_joined(
        dataframe: pd.DataFrame,
        base_dir_out: str,
        metrics_to_include: list = ["mse_mean"],
        show_plot: bool = False
):
    results_info = get_all_linear_results_joined(dataframe)

    for forecast_horizon, batch_data in results_info.items():
        plt.figure(figsize=(14, 8))
        for batch_size, lr_data in batch_data.items():
            for lr, lag_data in lr_data.items():
                ordered_lag_data = dict(sorted(lag_data.items(), key=lambda item: int(item[0])))
                lags = list(ordered_lag_data.keys())

                model_classes = list(next(iter(ordered_lag_data.values())).keys())

                for model_class in sorted(model_classes):
                    if "r2_mean" in metrics_to_include:
                        r2_values = [ordered_lag_data[lag][model_class]["r2_mean"] for lag in lags]
                        plt.plot(lags, r2_values, label=f"bs{batch_size}_lr{lr}_{model_class} - R2", linestyle=":", marker = 'o')

                    if "mae_mean" in metrics_to_include:
                        mae_values = [ordered_lag_data[lag][model_class]["mae_mean"] for lag in lags]
                        plt.plot(lags, mae_values, label=f"bs{batch_size}_lr{lr}_{model_class} - MAE", linestyle="--", marker = 'o')

                    if "mse_mean" in metrics_to_include:
                        mse_values = [ordered_lag_data[lag][model_class]["mse_mean"] for lag in lags]
                        plt.plot(lags, mse_values, label=f"bs{batch_size}_lr{lr}_{model_class}", linestyle="-", marker = 'o')

        plt.title(f"Forecast Horizon: {forecast_horizon}")
        plt.xlabel("Lag")
        plt.ylabel("MSE")
        plt.legend(title="Model Class and Metric")
        plt.grid(True)
        plt.tight_layout()

        if not os.path.exists(base_dir_out):
            os.makedirs(base_dir_out)

        plt.savefig(f"{base_dir_out}/fh{forecast_horizon}_results.png")
        if show_plot:
            plt.show()
        
        plt.clf()


def plot_linear_comparison_results(uncapped_df, capped_df, base_dir_out, metrics_to_include=["mse_mean"], show_plot=False):
    uncapped_results = get_best_linear_results(uncapped_df)
    capped_results = get_best_linear_results(capped_df)

    for forecast_horizon in uncapped_results.keys():

        fh_int = int(forecast_horizon)
        if fh_int not in capped_results:
            continue

        plt.figure(figsize=(14, 8))

        uncapped_lag_data = uncapped_results[forecast_horizon]
        ordered_uncapped = {int(k): v for k, v in sorted(uncapped_lag_data.items(), key=lambda item: int(item[0]))}
        uncapped_lags = list(ordered_uncapped.keys())

        capped_lag_data = capped_results[fh_int]
        ordered_capped = {int(k): v for k, v in sorted(capped_lag_data.items(), key=lambda item: int(item[0]))}
        capped_lags = list(ordered_capped.keys())

        common_lags = sorted(list(set(uncapped_lags) & set(capped_lags)))
        if not common_lags:
            plt.close()
            continue

        model_classes = list(next(iter(ordered_uncapped.values())).keys())

        for model_class in sorted(model_classes):
            if "r2_mean" in metrics_to_include:
                r2_values = [ordered_uncapped[lag][model_class]["r2_mean"] for lag in common_lags]
                plt.plot(common_lags, r2_values, label=f"Uncapped {model_class} - R2", linestyle=":", marker='o')

            if "mae_mean" in metrics_to_include:
                mae_values = [ordered_uncapped[lag][model_class]["mae_mean"] for lag in common_lags]
                plt.plot(common_lags, mae_values, label=f"Uncapped {model_class} - MAE", linestyle="--", marker='o')

            if "mse_mean" in metrics_to_include:
                mse_values = [ordered_uncapped[lag][model_class]["mse_mean"] for lag in common_lags]
                plt.plot(common_lags, mse_values, label=f"Uncapped {model_class}", linestyle="-", marker='o')

        for model_class in sorted(model_classes):
            if "r2_mean" in metrics_to_include:
                r2_values = [ordered_capped[lag][model_class]["r2_mean"] for lag in common_lags]
                plt.plot(common_lags, r2_values, label=f"Capped {model_class} - R2", linestyle=":", marker='x')

            if "mae_mean" in metrics_to_include:
                mae_values = [ordered_capped[lag][model_class]["mae_mean"] for lag in common_lags]
                plt.plot(common_lags, mae_values, label=f"Capped {model_class} - MAE", linestyle="--", marker='x')

            if "mse_mean" in metrics_to_include:
                mse_values = [ordered_capped[lag][model_class]["mse_mean"] for lag in common_lags]
                plt.plot(common_lags, mse_values, label=f"Capped {model_class}", linestyle="-", marker='x')

        plt.title(f"Forecast Horizon: {forecast_horizon} - Capped vs Uncapped Comparison")
        plt.xlabel("Lag")
        plt.ylabel("MSE")
        plt.legend(title="Data Type, Model Class and Metric")
        plt.grid(True)
        plt.tight_layout()

        if not os.path.exists(base_dir_out):
            os.makedirs(base_dir_out)

        plt.savefig(f"{base_dir_out}/comparison_fh{forecast_horizon}_results.png")
        if show_plot:
            plt.show()

        plt.close()


def get_all_var_results(dataframe: pd.DataFrame):
    results_info = {}
    model_class = "var"

    for r2, mse, mae, lag_forecast in zip(
      dataframe['r2_mean'], dataframe['mse_mean'],
      dataframe["mae_mean"], dataframe['config']
    ):
        lag_forecast = lag_forecast.split(",")
        lag_forecast = [value.strip("() ").strip() for value in lag_forecast]
        lag = int(lag_forecast[0])
        forecast_horizon = int(lag_forecast[1])
        results_info.setdefault(forecast_horizon, {})
        results_info[forecast_horizon].setdefault(lag, {})
        results_info[forecast_horizon][lag].setdefault(model_class, {})
        results_info[forecast_horizon][lag][model_class] = {
            "r2_mean": r2,
            "mae_mean": mae,
            "mse_mean": mse,
        }
    return results_info


def plot_var_comparison_results(uncapped_df, capped_df, base_dir_out, metrics_to_include=["mse_mean"], show_plot=False):
    uncapped_results = get_all_var_results(uncapped_df)
    capped_results = get_all_var_results(capped_df)
    
    for forecast_horizon in uncapped_results.keys():
        if forecast_horizon not in capped_results:
            continue
            
        plt.figure(figsize=(14, 8))
        
        uncapped_lag_data = uncapped_results[forecast_horizon]
        ordered_uncapped = dict(sorted(uncapped_lag_data.items(), key=lambda item: int(item[0])))
        lags = list(ordered_uncapped.keys())
        
        model_classes = list(next(iter(ordered_uncapped.values())).keys())
        
        for model_class in sorted(model_classes):
            if "r2_mean" in metrics_to_include:
                r2_values = [ordered_uncapped[lag][model_class]["r2_mean"] for lag in lags]
                plt.plot(lags, r2_values, label=f"Uncapped {model_class} - R2", linestyle=":", marker='o')
            
            if "mae_mean" in metrics_to_include:
                mae_values = [ordered_uncapped[lag][model_class]["mae_mean"] for lag in lags]
                plt.plot(lags, mae_values, label=f"Uncapped {model_class} - MAE", linestyle="--", marker='o')
            
            if "mse_mean" in metrics_to_include:
                mse_values = [ordered_uncapped[lag][model_class]["mse_mean"] for lag in lags]
                plt.plot(lags, mse_values, label=f"Uncapped {model_class}", linestyle="-", marker='o')
        
        # Plot capped results
        capped_lag_data = capped_results[forecast_horizon]
        ordered_capped = dict(sorted(capped_lag_data.items(), key=lambda item: int(item[0])))
        
        for model_class in sorted(model_classes):
            if "r2_mean" in metrics_to_include:
                r2_values = [ordered_capped[lag][model_class]["r2_mean"] for lag in lags]
                plt.plot(lags, r2_values, label=f"Capped {model_class} - R2", linestyle=":", marker='x')
            
            if "mae_mean" in metrics_to_include:
                mae_values = [ordered_capped[lag][model_class]["mae_mean"] for lag in lags]
                plt.plot(lags, mae_values, label=f"Capped {model_class} - MAE", linestyle="--", marker='x')
            
            if "mse_mean" in metrics_to_include:
                mse_values = [ordered_capped[lag][model_class]["mse_mean"] for lag in lags]
                plt.plot(lags, mse_values, label=f"Capped {model_class}", linestyle="-", marker='x')
        
        plt.title(f"VAR Model - Forecast Horizon: {forecast_horizon} - Capped vs Uncapped Comparison")
        plt.xlabel("Lag")
        plt.ylabel("MSE")
        plt.legend(title="Data Type, Model Class and Metric")
        plt.grid(True)
        plt.tight_layout()
        
        if not os.path.exists(base_dir_out):
            os.makedirs(base_dir_out)
        
        plt.savefig(f"{base_dir_out}/var_comparison_fh{forecast_horizon}_results.png")
        if show_plot:
            plt.show()
        
        plt.clf()


def plot_all_var_results(
        dataframe: pd.DataFrame,
        base_dir_out: str,
        metrics_to_include: list = ["mse_mean"],
        show_plot: bool = False
):
    results_info = get_all_var_results(dataframe)

    for forecast_horizon, lag_data in results_info.items():
        ordered_lag_data = dict(sorted(lag_data.items(), key=lambda item: int(item[0])))
        lags = list(ordered_lag_data.keys())
        model_classes = list(next(iter(ordered_lag_data.values())).keys())

        plt.figure(figsize=(14, 8))

        for model_class in sorted(model_classes):
            if "r2_mean" in metrics_to_include:
                r2_values = [
                    ordered_lag_data[lag][model_class]["r2_mean"] for lag in lags
                ]
                plt.plot(
                    lags, r2_values, label=f"{model_class} - R2",
                    linestyle=":", marker='o'
                )
            if "mae_mean" in metrics_to_include:
                mae_values = [
                    ordered_lag_data[lag][model_class]["mae_mean"] for lag in lags
                ]
                plt.plot(
                    lags, mae_values, label=f"{model_class} - MAE",
                    linestyle="--", marker='o'
                )
            if "mse_mean" in metrics_to_include:
                mse_values = [
                    ordered_lag_data[lag][model_class]["mse_mean"] for lag in lags
                ]
                plt.plot(
                    lags, mse_values, label=f"{model_class}",
                    linestyle="-", marker='o'
                )

        plt.title(f"Forecast Horizon: {forecast_horizon}")
        plt.xlabel("Lag")
        plt.ylabel("MSE")
        plt.legend(title="Model Class and Metric")
        plt.grid(True)
        plt.tight_layout()

        if not os.path.exists(base_dir_out):
            os.makedirs(base_dir_out)

        plt.savefig(f"{base_dir_out}/fh{forecast_horizon}_results.png")
        if show_plot:
            plt.show()
        
        plt.clf()


def plot_metrics(results, lags, classes, metrics_to_include, linestyle_map, prefix):
    for model_class in sorted(classes):
        for metric in metrics_to_include:
            metric_values = [results[lag][model_class][metric] for lag in lags]
            plt.plot(
                lags,
                metric_values,
                label=f"{prefix} {model_class} - {metric.upper()}",
                linestyle=linestyle_map[metric],
                marker='o'
            )


def plot_linear_var_patchtst_transf_results(
    dataframe_patchtst: pd.DataFrame,
    dataframe_linear: pd.DataFrame,
    dataframe_var: pd.DataFrame,
    dataframe_transf: pd.DataFrame,
    base_dir_out: str,
    metrics_to_include: list = ["mse_mean"],
    show_plot: bool = False
):
    results_patchtst = get_best_patchtst_results(dataframe_patchtst)
    results_linear = get_best_linear_results(dataframe_linear)
    results_var = get_all_var_results(dataframe_var)
    results_transf = get_best_transformer_results(dataframe_transf)

    os.makedirs(base_dir_out, exist_ok=True)

    linestyle_map = {"r2_mean": ":", "mae_mean": "--", "mse_mean": "-"}

    for lr, vr, pr, tr in zip(
        sorted(results_linear), sorted(results_var),
        sorted(results_patchtst), sorted(results_transf)
    ):
        plt.figure(figsize=(16, 10))        
        lr_res = results_linear[lr]
        vr_res = results_var[vr]
        ptst_res = results_patchtst[pr]
        transf_res = results_transf[tr]

        ordered_lr_res = {k: v for k, v in sorted(lr_res.items(), key=lambda item: int(item[0]))}
        ordered_vr_res = {k: v for k, v in sorted(vr_res.items(), key=lambda item: int(item[0]))}
        ordered_ptst_res = {k: v for k, v in sorted(ptst_res.items(), key=lambda item: int(item[0]))}
        ordered_transf_res = {k: v for k, v in sorted(transf_res.items(), key=lambda item: int(item[0]))}

        lr_lags = list(ordered_lr_res.keys())
        lr_classes = list(next(iter(ordered_lr_res.values())).keys())
        vr_lags = list(ordered_vr_res.keys())
        vr_classes = list(next(iter(ordered_vr_res.values())).keys())
        ptst_lags = list(ordered_ptst_res.keys())
        ptst_classes = list(next(iter(ordered_ptst_res.values())).keys())
        transf_lags = list(ordered_transf_res.keys())
        transf_classes = list(next(iter(ordered_transf_res.values())).keys())
        
        plot_metrics(ordered_lr_res, lr_lags, lr_classes, metrics_to_include, linestyle_map, "")
        plot_metrics(ordered_vr_res, vr_lags, vr_classes, metrics_to_include, linestyle_map, "")
        plot_metrics(ordered_ptst_res, ptst_lags, ptst_classes, metrics_to_include, linestyle_map, "")
        plot_metrics(ordered_transf_res, transf_lags, transf_classes, metrics_to_include, linestyle_map, "")

        forecast_horizon = lr
        plt.title(f"Forecast Horizon: {forecast_horizon}")
        plt.xlabel("Lag")
        plt.ylabel("MSE")
        plt.legend(
            title="Model Class and Metric",
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.,
            fontsize='small'
        )
        plt.grid(True)
        plt.tight_layout()

        plot_path = os.path.join(base_dir_out, f"fh{forecast_horizon}_results.png")
        plt.savefig(plot_path)

        if show_plot:
            plt.show()

        plt.clf()
