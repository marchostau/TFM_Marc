import os

import matplotlib.pyplot as plt
import pandas as pd


def get_config(row):
    config = row["config"]
    config = config.strip("{}")
    config = [x.split(": ") for x in config.split(", '")]
    config = [(x[0].replace("'", ''), x[1]) for x in config]
    config = dict(config)
    return config


def get_lag(row):
    config = get_config(row)
    lag_forecast = config["lag_forecast"].split(",")
    lag_forecast = [value.strip("[] ").strip() for value in lag_forecast]
    return lag_forecast[0]


def get_forecast_horizon(row):
    config = get_config(row)
    lag_forecast = config["lag_forecast"].split(",")
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


def get_best_linear_results(dataframe: pd.DataFrame, min_by: str = 'mse'):
    dataframe["lag"] = df.apply(get_lag, axis=1)
    dataframe["forecast_horizon"] = df.apply(get_forecast_horizon, axis=1)
    dataframe["model_class"] = df.apply(get_model_class, axis=1)
    dataframe["batch_size"] = df.apply(get_batch_size, axis=1)
    dataframe["lr"] = df.apply(get_learning_rate, axis=1)

    print(f"Initial df:\n{dataframe}")
    
    """
    grouped_df = dataframe.groupby(
        ['lag', 'forecast_horizon', 'model_class'],
        as_index=False
    ).agg(
        min_mse=('mse', 'min'),
        min_mae=('mae', 'min'),
        min_r2=('r2', 'min')
    )"""
    # .reset_index()

    grouped_df = dataframe.loc[
        df.groupby(
            ['lag', 'forecast_horizon', 'model_class']
        )[min_by].idxmin()
    ].reset_index()

    print(f"GDF1):\n{grouped_df}")


    results_info = {}
    for lag, forecast_horizon, model_class, mse, mae, r2, batch_size, lr in zip(
        grouped_df['lag'], grouped_df['forecast_horizon'],
        grouped_df['model_class'], grouped_df['mse'],
        grouped_df['mae'], grouped_df['r2'],
        grouped_df['batch_size'], grouped_df['lr']
    ):
        results_info.setdefault(forecast_horizon, {})
        results_info[forecast_horizon].setdefault(lag, {})
        results_info[forecast_horizon][lag].setdefault(model_class, {})
        results_info[forecast_horizon][lag][model_class] = {
            "r2": r2,
            "mae": mae,
            "mse": mse,
            "batch_size": batch_size,
            "lr": lr
        }
    return results_info


def get_all_linear_results(dataframe: pd.DataFrame):
    results_info = {}

    for config, r2, mae, mse in zip(
        dataframe['config'], dataframe['r2'],
        dataframe['mae'], dataframe['mse']
    ):
        config = config.strip("{}")
        config = [x.split(": ") for x in config.split(", '")]
        config = [(x[0].replace("'", ''), x[1]) for x in config]
        config = dict(config)

        model_class = str(config["model_class"]).split(
            '.'
        )[-1].replace("'>", "")
        batch_size = config["batch_size"]
        lr = config["lr"]

        lag_forecast = config["lag_forecast"].split(",")
        lag_forecast = [value.strip("[] ").strip() for value in lag_forecast]
        lag = lag_forecast[0]
        forecast_horizon = lag_forecast[1]

        results_info.setdefault(batch_size, {})
        results_info[batch_size].setdefault(lr, {})
        results_info[batch_size][lr].setdefault(forecast_horizon, {})
        results_info[batch_size][lr][forecast_horizon].setdefault(lag, {})
        results_info[batch_size][lr][forecast_horizon][lag][model_class] = {
            "r2": r2,
            "mae": mae,
            "mse": mse
        }

    return results_info


def plot_best_linear_results(
        dataframe: pd.DataFrame,
        base_dir_out: str,
        metrics_to_include: list = ["mse"],
        show_plot: bool = False
):
    results_info = get_best_linear_results(dataframe)

    for forecast_horizon, lag_data in results_info.items():
        ordered_lag_data = dict(sorted(lag_data.items(), key=lambda item: int(item[0])))
        lags = list(ordered_lag_data.keys())

        model_classes = list(next(iter(ordered_lag_data.values())).keys())

        plt.figure(figsize=(14, 8))

        for model_class in sorted(model_classes):
            if "r2" in metrics_to_include:
                r2_values = [
                    ordered_lag_data[lag][model_class]["r2"] for lag in lags
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
            if "mae" in metrics_to_include:
                mae_values = [
                    ordered_lag_data[lag][model_class]["mae"] for lag in lags
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
            if "mse" in metrics_to_include:
                mse_values = [
                    ordered_lag_data[lag][model_class]["mse"] for lag in lags
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


def plot_all_linear_results(
        dataframe: pd.DataFrame,
        base_dir_out: str,
        metrics_to_include: list = ["mse"],
        show_plot: bool = False
):
    results_info = get_all_linear_results(dataframe)

    for batch_size, lr_data in results_info.items():
        for lr, forecast_data in lr_data.items():
            for forecast_horizon, lag_data in forecast_data.items():
                ordered_lag_data = dict(sorted(lag_data.items(), key=lambda item: int(item[0])))
                lags = list(ordered_lag_data.keys())

                model_classes = list(next(iter(ordered_lag_data.values())).keys())

                plt.figure(figsize=(14, 8))

                for model_class in sorted(model_classes):
                    if "r2" in metrics_to_include:
                        r2_values = [ordered_lag_data[lag][model_class]["r2"] for lag in lags]
                        plt.plot(lags, r2_values, label=f"{model_class} - R2", linestyle=":", marker = 'o')

                    if "mae" in metrics_to_include:
                        mae_values = [ordered_lag_data[lag][model_class]["mae"] for lag in lags]
                        plt.plot(lags, mae_values, label=f"{model_class} - MAE", linestyle="--", marker = 'o')

                    if "mse" in metrics_to_include:
                        mse_values = [ordered_lag_data[lag][model_class]["mse"] for lag in lags]
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

"""
results_path = (
    '/home/marchostau/Desktop/TFM/Code/ProjectCode/'
    'models/evaluate_results/linear_models/results'
    '[(3,3),(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)].csv'
)
df = pd.read_csv(results_path)

# base_dir_out = '/home/marchostau/Desktop/TFM/Code/ProjectCode/models/plots/testing_results/linear_models/all_results'
# plot_all_linear_results(df, base_dir_out)
# get_best_linear_results(df)

base_dir_out = '/home/marchostau/Desktop/TFM/Code/ProjectCode/models/plots/testing_results/linear_models/best_results'
plot_best_linear_results(df, base_dir_out)
"""