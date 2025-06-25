# Wind Forecasting for Sailing Races using Statistical and Deep Learning Models

## Project Overview

This project aims to forecast short- to medium-term wind behavior to support tactical decision-making in sailboat races. It combines classical statistical models with modern deep learning architectures (Linear, NLinear, Transformer Encoder, PatchTST), enabling a comparative evaluation of forecasting techniques using high-resolution wind data.

---
```bash
## Project Structure
ProjectCode/
├── config.yaml # Main configuration file
├── main.py # Entry point for data processing, training and evaluation
├── requirements.txt # Python dependencies

├── data_processing/ # Data preprocessing utilities
│ ├── config_schema.py # Config validation schema for reading raw data
│ ├── file_loader.py # File reading logic. Used for reading the days data in Texys Marine frame format. It's used to obtain the main variables of the raw days data, obtaining wind speed, wind direction, u and v components, latitude and longitude...
│ ├── process_data.py # Data cleaning, slicing, formatting. Once the raw data is readed, the days in csv format are preprocessed
│ ├── dataset_loader.py # Custom PyTorch Dataset for loading the u and v components of the segments
│ ├── normalization.py # Normalization logic used by the process_data.py file
│ ├── utils.py # Generic preprocessing utilities 
│ └── visualization.py # Data plots and insights of the data (distribution plots, evolution plots, computing adfuller tests...)

├── logging_information/ # Logging and debug configuration
│ ├── init.py
│ ├── logging_config.py # Logging setup
│ └── logs/ # Stored logs

├── models/ # Forecasting models
│ ├── var_model.py # Vector AutoRegressive (VAR) model
│ ├── linear_models.py # Linear and NonLinear regression models
│ ├── transformer_encoder.py # Encoder-only Transformer
│ ├── patch_tst.py # PatchTST Transformer model
│ ├── utils.py # Training and evaluation helpers
│ ├── visualization.py # Forecast visualizations
│ └── init.py
```

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure the Pipeline
First of all, load all the raw days data with load_dir_text method inside the file_loader.py file, in order to extract the interesting variables from the raw days data (In Texys Marine frame format). Then edit the config.yaml, configuring where the days in csv format are situated and the output directory. In this configuration file you have to define which preprocessing methods and which techniques you want to use. Finally, you can run the run_experiments methods of each one of the models with the segments in csv format obtained after the preprocessing. In the run_experiments methods you can specify which hyperparameters you want to use for each model and which configurations of lag and forecast horizon you desire to use.

## Models Implemented
### VAR (Vector AutoRegressive)
Classical multivariate statistical model effective for stationary data with strong inter-variable dependencies (e.g., wind U and V components).

### Linear / NLinear
Lightweight baseline models from recent literature. Despite simplicity, they show competitive performance in some tasks (e.g., Zeng et al. 2022 - *Are Transformers Effective for Time Series Forecasting?*).

### Transformer Encoder
Encoder-only architecture leveraging attention mechanisms for learning temporal dependencies without relying on recurrence.

### PatchTST
Transformer model using patch-based time series encoding and channel-independence strategy.

