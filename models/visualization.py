import matplotlib.pyplot as plt
import pandas as pd


def plot_results(dataframe: pd.DataFrame):
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

    for batch_size, lr_data in results_info.items():
        for lr, forecast_data in lr_data.items():
            for forecast_horizon, lag_data in forecast_data.items():
                ordered_lag_data = dict(sorted(lag_data.items(), key=lambda item: int(item[0])))
                lags = list(ordered_lag_data.keys())

                model_classes = list(next(iter(ordered_lag_data.values())).keys())

                plt.figure(figsize=(14, 8))

                for model_class in model_classes:
                    r2_values = [ordered_lag_data[lag][model_class]["r2"] for lag in lags]
                    mae_values = [ordered_lag_data[lag][model_class]["mae"] for lag in lags]
                    mse_values = [ordered_lag_data[lag][model_class]["mse"] for lag in lags]

                    print(f"MSE values for batch_size {batch_size}, lr {lr}, f_h {forecast_horizon}, lag {lag} model {model_class}\n{mse_values}")

                    plt.plot(lags, r2_values, label=f"{model_class} - R2", linestyle="-", marker = 'o')
                    plt.plot(lags, mae_values, label=f"{model_class} - MAE", linestyle="--", marker = 'o')
                    plt.plot(lags, mse_values, label=f"{model_class} - MSE", linestyle=":", marker = 'o')

                plt.title(f"Batch Size: {batch_size}, LR: {lr}, Forecast Horizon: {forecast_horizon}")
                plt.xlabel("Lag")
                plt.ylabel("Metrics")
                plt.legend(title="Model Class and Metric")
                plt.grid(True)
                plt.tight_layout()

                plt.savefig(f"results_bs{batch_size}_lr{lr}_fh{forecast_horizon}.png")
                #plt.show()


results_path = (
    '/home/marchostau/Desktop/TFM/Code/ProjectCode/'
    'models/evaluate_results/linear_models/results'
    '[(3,3),(6,6),(9,9),(12,12),(6,3),(9,3),(9,6),(12,6),(12,9)].csv'
)
df = pd.read_csv(results_path)
plot_results(df)
