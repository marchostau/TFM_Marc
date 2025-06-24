import os
import glob

import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import matplotlib.dates as mdates
from ..models.utils import concatenate_datasets 
from matplotlib.ticker import MaxNLocator
import seaborn as sns

from ..logging_information.logging_config import get_logger

logger = get_logger(__name__)


def compute_adfuller_test(df: pd.DataFrame, variable: str, minimum: int = 50) -> dict:
    """Run ADF test if sample size >=50; return results as a dict."""
    results = {
        'adf_statistic': None,
        'p_value': None,
        'is_stationary': None,
        'sample_size': len(df),
        'can_test': len(df) >= minimum
    }
    
    if results['can_test']:
        try:
            adf_result = adfuller(df[variable].dropna())  # Handle NaN values
            results['adf_statistic'] = adf_result[0]
            results['p_value'] = adf_result[1]
            results['is_stationary'] = adf_result[1] <= 0.05
        except Exception as e:
            logger.error(f"Error in ADF test for {variable}: {str(e)}")
            results['can_test'] = False
    return results


def analyze_data_directory(dir_source: str, variable: str, output_dir: str) -> dict:
    """Analyze all CSV files in a directory, returning aggregated stats.
    Only plots testable days (≥50 timesteps) in stationarity-specific folders."""
    file_list = sorted(glob.glob(os.path.join(dir_source, "*.csv")))
    summary = {
        'total_days': len(file_list),
        'testable_days': 0,
        'stationary_days': 0,
        'non_stationary_days': 0,
        'skipped_days': 0
    }
    
    # Create output directories
    stat_dir = os.path.join(output_dir, f"stationary_{variable}")
    non_stat_dir = os.path.join(output_dir, f"non_stationary_{variable}")
    os.makedirs(stat_dir, exist_ok=True)
    os.makedirs(non_stat_dir, exist_ok=True)

    for file_path in file_list:
        try:
            df = pd.read_csv(file_path, parse_dates=["timestamp"])
            if variable not in df.columns:
                summary['skipped_days'] += 1
                continue
                
            # Skip if not testable (<50 timesteps)
            if len(df) < 50:
                summary['skipped_days'] += 1
                continue
                
            # Determine stationarity
            results = compute_adfuller_test(df, variable)
            summary['testable_days'] += 1
            
            # Extract filename without extension
            filename = os.path.splitext(os.path.basename(file_path))[0]
            
            # Create plot path based on stationarity
            if results['is_stationary']:
                summary['stationary_days'] += 1
                output_path = os.path.join(stat_dir, f"{filename}.png")
            else:
                summary['non_stationary_days'] += 1
                output_path = os.path.join(non_stat_dir, f"{filename}.png")
            
            # Create the plot
            plt.figure(figsize=(12, 4))
            plt.plot(df['timestamp'], df[variable], color='steelblue', linewidth=1)
            
            # Format x-axis to show only hours:minutes
            ax = plt.gca()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            # Set informative title
            stationarity_status = "STATIONARY" if results['is_stationary'] else "NON-STATIONARY"
            plt.title(f"{variable.upper()} - {stationarity_status}\n{filename}")
            
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path, dpi=100)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            summary['skipped_days'] += 1
    
    return summary


def print_summary(stats: dict, variable: str):
    """Log a formatted summary of results."""
    logger.info(f"\n=== Summary for {variable.upper()} ===")
    logger.info(f"Total days: {stats['total_days']}")
    logger.info(f"Days testable (≥50 timesteps): {stats['testable_days']}")
    logger.info(f"  → Stationary days: {stats['stationary_days']}")
    logger.info(f"  → Non-stationary days: {stats['non_stationary_days']}")
    logger.info(f"Skipped days (<50 timesteps or errors): {stats['skipped_days']}")


def extract_date_from_filename(file_path: str) -> str:
    """Extract date from filenames like '2022-05-12__aut_ivan.txt_processed_segment_1.csv'."""
    filename = os.path.basename(file_path)
    date_part = filename.split('_')[0]  # Get "2022-05-12"
    try:
        datetime.strptime(date_part, "%Y-%m-%d")  # Validate date format
        return date_part
    except ValueError:
        logger.warning(f"Invalid date format in filename: {filename}")
        return None


def analyze_timestamps_and_days(dir_source: str, output_dir: str) -> dict:
    """Analyze timestamps per day and count unique days."""
    file_list = glob.glob(os.path.join(dir_source, "*.csv"))
    date_counts = {}
    timestamps_per_day = {}
    
    for file_path in file_list:
        date = extract_date_from_filename(file_path)
        if not date:
            continue
        
        date_counts[date] = date_counts.get(date, 0) + 1
        
        try:
            df = pd.read_csv(file_path, parse_dates=["timestamp"])
            if 'timestamp' not in df.columns:
                logger.warning(f"No 'timestamp' column in {file_path}")
                continue
            
            df['date'] = df['timestamp'].dt.date
            if date not in timestamps_per_day:
                timestamps_per_day[date] = []
            timestamps_per_day[date].extend(df['timestamp'].tolist())
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    timestamps_per_day_count = {date: len(timestamps) for date, timestamps in timestamps_per_day.items()}
    
    # Save histogram and get bin stats
    distribution_path = os.path.join(output_dir, "timesteps_distribution.png")
    bin_stats = plot_timesteps_distribution(timestamps_per_day_count, distribution_path)
    
    return {
        'unique_days': list(date_counts.keys()),
        'files_per_day': date_counts,
        'timestamps_per_day': timestamps_per_day_count,
        'bin_stats': bin_stats  # Add bin statistics to return dict
    }


def analyze_segments_distribution(dir_source: str, output_dir: str) -> dict:
    """Analyze timesteps distribution across all segments (not grouped by day)"""
    file_list = glob.glob(os.path.join(dir_source, "*.csv"))
    segment_lengths = []
    
    for file_path in file_list:
        try:
            df = pd.read_csv(file_path)
            segment_lengths.append(len(df))
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    # Define bin edges
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    
    # Plot histogram
    plt.figure(figsize=(12, 6))
    counts, bin_edges, _ = plt.hist(segment_lengths, 
                                  bins=bins,
                                  color='skyblue', 
                                  edgecolor='black')
    
    # Calculate statistics
    bin_ranges = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)]
    total_segments = len(segment_lengths)
    
    logger.info("\n=== Segment Length Distribution ===")
    logger.info(f"{'Bin Range':<12} | {'Segments':<8} | {'Percentage':<8}")
    logger.info("-"*35)
    for rng, cnt in zip(bin_ranges, counts):
        pct = (cnt/total_segments)*100
        logger.info(f"{rng:<12} | {int(cnt):<8} | {pct:.1f}%")
    
    # Format plot
    plt.xticks(bins, rotation=45)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Number of Timesteps per Segment')
    plt.ylabel('Number of Segments')
    plt.title('Distribution of Segment Lengths')
    plt.grid(axis='y', alpha=0.3)
    
    # Save plot
    output_path = os.path.join(output_dir, "segment_lengths_distribution.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved segment lengths distribution plot to {output_path}")
    
    return {
        'bin_ranges': bin_ranges,
        'bin_counts': counts.tolist(),
        'total_segments': total_segments,
        'min_length': min(segment_lengths),
        'max_length': max(segment_lengths),
        'median_length': np.median(segment_lengths)
    }


def save_plot(data: pd.Series, title: str, ylabel: str, output_path: str):
    """Generic function to save a time series plot."""
    plt.figure(figsize=(12, 6))
    plt.plot(data, color='blue')
    plt.title(title)
    plt.xlabel('Timestamp')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved plot to {output_path}")


def plot_timesteps_distribution(timestamps_per_day: dict, output_path: str):
    """
    Plot histogram with exact integer bin ranges: 
    0-20-40-60-80-100-120-140-160-180-200-220-240
    Returns and prints detailed bin counts
    """
    counts = list(timestamps_per_day.values())
    
    if not counts:
        logger.warning("No timestamp data available for distribution plot")
        return None
    
    # Define exact bin edges as specified (integers)
    bins = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
    
    plt.figure(figsize=(12, 6))
    hist, bin_edges, _ = plt.hist(counts, 
                                bins=bins,
                                color='skyblue', 
                                edgecolor='black')
    
    # Calculate and print bin statistics
    bin_ranges = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)]
    bin_counts = hist.astype(int)
    total_segments = sum(bin_counts)
    
    logger.info("\n=== Segment Count by Timestep Range ===")
    logger.info(f"{'Bin Range':<12} | {'Segments':<8} | {'Percentage':<8}")
    logger.info("-"*35)
    for rng, cnt in zip(bin_ranges, bin_counts):
        pct = (cnt/total_segments)*100
        logger.info(f"{rng:<12} | {cnt:<8} | {pct:.1f}%")
    
    # Set x-axis ticks at bin edges (integer values)
    plt.xticks(bins, rotation=45)
    
    # Force y-axis to use integer values only
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Number of Segments')
    plt.title('Segment Timestep Count Distribution')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved timesteps distribution plot to {output_path}")
    
    return {
        'bin_ranges': bin_ranges,
        'bin_counts': bin_counts.tolist(),
        'total_segments': total_segments
    }


def plot_component_time_series(df: pd.DataFrame, component: str, output_dir: str):
    """Plot time series showing 4 months per year on x-axis"""
    output_path = os.path.join(output_dir, f"{component}_time_series.png")
    
    plt.figure(figsize=(14, 5))
    plt.plot(df['timestamp'], df[component], color='steelblue', linewidth=0.8)
    
    # Configure x-axis to show 4 months per year
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show every 3rd month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))  # Format as "Jan\n2022"
    
    plt.title(f'{component.upper()} Component Time Series')
    plt.xlabel('Time (Month/Year)')
    plt.ylabel(f'{component.upper()} Value')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved {component} time series plot to {output_path}")


def print_concatenated_stats(dir_source: str):
    """Calculate and print mean/std of u and v components from concatenated data"""
    try:
        # Load concatenated dataset
        df_combined = concatenate_datasets(dir_source)
        
        # Calculate statistics
        u_mean = df_combined['u_component'].mean()
        u_std = df_combined['u_component'].std()
        v_mean = df_combined['v_component'].mean()
        v_std = df_combined['v_component'].std()
        
        # Print results
        logger.info("\n=== Concatenated Dataset Statistics ===")
        logger.info(f"Total timesteps: {len(df_combined)}")
        logger.info(f"U Component - Mean: {u_mean:.4f} m/s, Std: {u_std:.4f} m/s")
        logger.info(f"V Component - Mean: {v_mean:.4f} m/s, Std: {v_std:.4f} m/s")
        
        # Additional statistics
        logger.info("\nAdditional Statistics:")
        logger.info(f"U Component - Min: {df_combined['u_component'].min():.4f} m/s")
        logger.info(f"U Component - Max: {df_combined['u_component'].max():.4f} m/s")
        logger.info(f"V Component - Min: {df_combined['v_component'].min():.4f} m/s")
        logger.info(f"V Component - Max: {df_combined['v_component'].max():.4f} m/s")
        
        # Return values in case you want to use them elsewhere
        return {
            'u_mean': u_mean,
            'u_std': u_std,
            'v_mean': v_mean,
            'v_std': v_std,
            'total_samples': len(df_combined)
        }
        
    except Exception as e:
        logger.info(f"Error calculating statistics: {str(e)}")
        return None