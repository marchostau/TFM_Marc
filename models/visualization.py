import os

import matplotlib.pyplot as plt
import pandas as pd

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


def plot_comparison_best_results(uncapped_df, capped_df, base_dir_out, metrics_to_include=["mse_mean"], show_plot=False):
    # Get best results from both datasets
    uncapped_results = get_best_linear_results(uncapped_df)
    capped_results = get_best_linear_results(capped_df)
    
    # Plot comparison for each forecast horizon
    for forecast_horizon in uncapped_results.keys():
        if forecast_horizon not in capped_results:
            continue
            
        plt.figure(figsize=(14, 8))
        
        # Plot uncapped results
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
                plt.plot(lags, mse_values, label=f"Uncapped {model_class} - MSE", linestyle="-", marker='o')
        
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
                plt.plot(lags, mse_values, label=f"Capped {model_class} - MSE", linestyle="-", marker='x')
        
        plt.title(f"Forecast Horizon: {forecast_horizon} - Capped vs Uncapped Comparison")
        plt.xlabel("Lag")
        plt.ylabel("Metrics")
        plt.legend(title="Data Type, Model Class and Metric")
        plt.grid(True)
        plt.tight_layout()
        
        if not os.path.exists(base_dir_out):
            os.makedirs(base_dir_out)
        
        plt.savefig(f"{base_dir_out}/comparison_fh{forecast_horizon}_results.png")
        if show_plot:
            plt.show()
        
        plt.clf()


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
                    f"LR: {ordered_lag_data[lag][model_class]['lr']}) - MSE"
                    for lag in lags
                ]
                plt.plot(
                    lags, mse_values, label=", ".join(config_labels),
                    linestyle="-", marker='o'
                )

        plt.title(f"Forecast Horizon: {forecast_horizon}")
        plt.xlabel("Lag")
        plt.ylabel("Metrics")
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
                        plt.plot(lags, mse_values, label=f"{model_class} - MSE", linestyle="-", marker = 'o')

                plt.title(f"Batch Size: {batch_size}, LR: {lr}, Forecast Horizon: {forecast_horizon}")
                plt.xlabel("Lag")
                plt.ylabel("Metrics")
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
                        plt.plot(lags, mse_values, label=f"bs{batch_size}_lr{lr}_{model_class} - MSE", linestyle="-", marker = 'o')

        plt.title(f"Forecast Horizon: {forecast_horizon}")
        plt.xlabel("Lag")
        plt.ylabel("Metrics")
        plt.legend(title="Model Class and Metric")
        plt.grid(True)
        plt.tight_layout()

        if not os.path.exists(base_dir_out):
            os.makedirs(base_dir_out)

        plt.savefig(f"{base_dir_out}/fh{forecast_horizon}_results.png")
        if show_plot:
            plt.show()
        
        plt.clf()


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
    # Get results from both datasets
    uncapped_results = get_all_var_results(uncapped_df)
    capped_results = get_all_var_results(capped_df)
    
    # Plot comparison for each forecast horizon
    for forecast_horizon in uncapped_results.keys():
        if forecast_horizon not in capped_results:
            continue
            
        plt.figure(figsize=(14, 8))
        
        # Plot uncapped results
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
                plt.plot(lags, mse_values, label=f"Uncapped {model_class} - MSE", linestyle="-", marker='o')
        
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
                plt.plot(lags, mse_values, label=f"Capped {model_class} - MSE", linestyle="-", marker='x')
        
        plt.title(f"VAR Model - Forecast Horizon: {forecast_horizon} - Capped vs Uncapped Comparison")
        plt.xlabel("Lag")
        plt.ylabel("Metrics")
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
                    lags, mse_values, label=f"{model_class} - MSE",
                    linestyle="-", marker='o'
                )

        plt.title(f"Forecast Horizon: {forecast_horizon}")
        plt.xlabel("Lag")
        plt.ylabel("Metrics")
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


def plot_linear_and_var_results(
    dataframe_linear: pd.DataFrame,
    dataframe_var: pd.DataFrame,
    base_dir_out: str,
    metrics_to_include: list = ["mse_mean"],
    show_plot: bool = False
):
    results_linear = get_best_linear_results(dataframe_linear)
    results_var = get_all_var_results(dataframe_var)

    print(f"Results linear: {results_linear}")
    print(f"Results var: {results_var}")

    os.makedirs(base_dir_out, exist_ok=True)

    linestyle_map = {"r2_mean": ":", "mae_mean": "--", "mse_mean": "-"}

    for lr, vr in zip(sorted(results_linear), sorted(results_var)):

        print(f"LR: {lr}")
        print(f"VR: {vr}")
        
        lr_res = results_linear[lr]
        vr_res = results_var[vr]

        ordered_lr_res = {k: v for k, v in sorted(lr_res.items(), key=lambda item: int(item[0]))}
        ordered_vr_res = {k: v for k, v in sorted(vr_res.items(), key=lambda item: int(item[0]))}

        lr_lags = list(ordered_lr_res.keys())
        lr_classes = list(next(iter(ordered_lr_res.values())).keys())
        vr_lags = list(ordered_vr_res.keys())
        vr_classes = list(next(iter(ordered_vr_res.values())).keys())

        plot_metrics(ordered_lr_res, lr_lags, lr_classes, metrics_to_include, linestyle_map, "")
        plot_metrics(ordered_vr_res, vr_lags, vr_classes, metrics_to_include, linestyle_map, "")

        forecast_horizon = lr
        plt.title(f"Forecast Horizon: {forecast_horizon}")
        plt.xlabel("Lag")
        plt.ylabel("Metrics")
        plt.legend(title="Model Class and Metric")
        plt.grid(True)
        plt.tight_layout()

        plot_path = os.path.join(base_dir_out, f"fh{forecast_horizon}_results.png")
        plt.savefig(plot_path)

        if show_plot:
            plt.show()

        plt.clf()


base_dir = "/home/marchostau/Desktop/TFM/Code/ProjectCode/models"
results_suffix = (
    "results[((3,3),(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]_capped_data"
)
results_path = f"{base_dir}/evaluate_results/linear_models/{results_suffix}/AllResults/results_averaged.csv"
#results_path = f"{base_dir}/evaluate_results/linear_models/{results_suffix}/AllResults/testing_results_seedNone.csv"

df_linear = pd.read_csv(results_path)

plot_base = f"{base_dir}/plots/testing_results/linear_models/{results_suffix}"

plot_all_linear_results_joined(df_linear, f"{plot_base}/joined_results")
plot_all_linear_results_separated(df_linear, f"{plot_base}/separated_results")
plot_best_linear_results(df_linear, f"{plot_base}/best_results")

results_path = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/models/"
    "evaluate_results/var_model/results[((3,3),(6,6),(9,9),"
    "(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]_capped_data/AllResults/"
    "results_averaged.csv"
)
df_var = pd.read_csv(results_path)

base_dir_out = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/"
    "models/plots/testing_results/var_model/results"
    "[((3,3),(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),"
    "(12,6),(12,9)]_capped_data"
)
#plot_all_var_results(df_var, base_dir_out)

base_dir_out = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/models/"
    "plots/testing_results/linear_vs_var/results[((3,3),(6,6),(9,9),"
    "(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]_capped_data"
)
plot_linear_and_var_results(df_linear, df_var, base_dir_out)


base_dir = "/home/marchostau/Desktop/TFM/Code/ProjectCode/models"

# Uncapped data
uncapped_suffix = "results[((3,3),(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]"
uncapped_path = f"{base_dir}/evaluate_results/linear_models/{uncapped_suffix}/AllResults/results_averaged.csv"
df_uncapped = pd.read_csv(uncapped_path)

# Capped data
capped_suffix = "results[((3,3),(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]_capped_data"
capped_path = f"{base_dir}/evaluate_results/linear_models/{capped_suffix}/AllResults/results_averaged.csv"
df_capped = pd.read_csv(capped_path)

base_dir_out = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/models/"
    "plots/testing_results/linear_models/results[((3,3),(6,6),(9,9),"
    "(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]_cappedVSuncapped"
)

plot_comparison_best_results(
    df_uncapped,
    df_capped,
    base_dir_out,
    metrics_to_include=["mse_mean"]
)


"""
base_dir = "/home/marchostau/Desktop/TFM/Code/ProjectCode/models"

# Uncapped VAR data
uncapped_var_suffix = "results[((3,3),(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]"
uncapped_var_path = f"{base_dir}/evaluate_results/var_model/{uncapped_var_suffix}/AllResults/results_averaged.csv"
df_var_uncapped = pd.read_csv(uncapped_var_path)

# Capped VAR data
capped_var_suffix = "results[((3,3),(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]_capped_data"
capped_var_path = f"{base_dir}/evaluate_results/var_model/{capped_var_suffix}/AllResults/results_averaged.csv"
df_var_capped = pd.read_csv(capped_var_path)

base_dir_out = (
    "/home/marchostau/Desktop/TFM/Code/ProjectCode/models/"
    "plots/testing_results/var_model/results[((3,3),(6,6),(9,9),"
    "(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)]_cappedVSuncapped"
)

plot_var_comparison_results(
    df_var_uncapped,
    df_var_capped,
    base_dir_out
)
"""