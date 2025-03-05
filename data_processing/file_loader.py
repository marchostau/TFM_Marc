import os

import pandas as pd
from metpy import calc
from metpy.units import units

from ..logging_information.logging_config import get_logger

logger = get_logger(__name__)


def compute_wind_speed(u_component: float, v_component: float):
    return calc.wind_speed(
        u_component * units('m/s'), v_component * units('m/s')
    )


def compute_wind_direction(u_component: float, v_component: float):
    return calc.wind_direction(
        u_component * units('m/s'), v_component * units('m/s')
    )


def compute_u_v_comp(wind_speed: float, meteo_wind_dir: float):
    components = calc.wind_components(
        wind_speed * units('m/s'), meteo_wind_dir * units.deg
    )
    return components[0].m.item(), components[1].m.item()


def parse_line(line: str):
    parts = line.split(',')

    if len(parts) < 19:
        logger.warning(
            "Skipping line due to insufficient data fields. "
            f"Expected at least 19, found {len(parts)}."
        )
        return None

    date_str = parts[2]
    time_str = parts[3]
    latitude = parts[4]
    longitude = parts[6]
    wind_speed = parts[9]
    wind_direction = parts[10]
    wind_flag = parts[-1]

    if any(
        value == '' or value == '\n'
        for value in [
            date_str, time_str, latitude, longitude,
            wind_speed, wind_direction, wind_flag
        ]
    ):
        logger.warning(
            f"Skipping line due to missing or malformed values. "
            f"Date: {date_str}, Time: {time_str}, Lat: {latitude}, "
            f"Long: {longitude}, "
            f"Wind Speed: {wind_speed}, Wind Direction: {wind_direction}, "
            f"Wind Flag: {wind_flag}"
        )
        return None

    if date_str == '00':
        logger.warning(f"Ignoring entry with invalid date: {date_str}")
        return None

    try:
        latitude = float(latitude)
        longitude = float(longitude)
        wind_speed = float(wind_speed) * 0.514444  # Convert knots to m/s
        wind_direction = float(wind_direction)
        wind_flag = int(wind_flag.split('*')[0])
    except ValueError:
        logger.warning(
            "Data conversion error: Unable to parse "
            f"numerical values in line: {line}",
            exc_info=True
        )
        return None

    if wind_flag != 0:
        return None

    try:
        timestamp = pd.to_datetime(
            date_str + time_str, format='%d%m%y%H%M%S.%f'
        )
    except ValueError:
        logger.warning(
            f"Timestamp conversion failed for Date: {date_str}, "
            f"Time: {time_str}. Skipping entry.",
            exc_info=True
        )
        return None

    u_component, v_component = compute_u_v_comp(wind_speed, wind_direction)

    return {
        'timestamp': timestamp,
        'latitude': latitude,
        'longitude': longitude,
        'wind_speed': wind_speed,
        'wind_direction': wind_direction,
        'wind_flag': wind_flag,
        'u_component': u_component,
        'v_component': v_component
    }


def load_txt_file(file_path: str, filename: str):
    data = []
    if os.path.isfile(file_path):
        logger.info(f"Starting processing of file: {filename}")
        with open(file_path, 'r') as f:
            file_content = f.readlines()
            for line in file_content:
                parsed_data = parse_line(line)
                if parsed_data:
                    parsed_data['file_name'] = filename
                    data.append(parsed_data)
    return pd.DataFrame(data) if data else pd.DataFrame()


def load_dir_txt(dir_source: str, dir_output: str):
    if not os.path.isdir(dir_source):
        logger.error(
            f"Invalid directory: {dir_source}. "
            "Please provide a valid source directory."
        )
        raise ValueError('{dir_source} is not a directory')

    files = sorted(os.listdir(dir_source))
    for filename in files:
        file_path = os.path.join(dir_source, filename)
        dataframe = load_txt_file(file_path, filename)

        if not dataframe.empty:
            output_file = os.path.join(dir_output, f"{filename}.csv")
            dataframe.to_csv(output_file, index=False)
            logger.info(
                f"File loaded successfully and saved as: {output_file}"
            )
        else:
            logger.warning(
                f"No valid data extracted from {filename}. "
                "No output file generated."
            )
