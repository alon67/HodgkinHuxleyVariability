# replot_cli.py
"""
CLI entrypoint for replotting figures from saved HH sensitivity analysis data.
Replicates the functionality of the original replot_figures.py script.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, Any
import config
import parameters
import plotting


def check_required_files(data_dir: str = '.') -> bool:
    """Check if all required data files exist in the specified directory."""
    required_files = [
        os.path.join(data_dir, config.TIME_VECTOR_CSV),
        os.path.join(data_dir, config.PARAM_NAMES_TXT),
        os.path.join(data_dir, config.MEAN_V_CSV)
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"ERROR: Missing required files: {missing_files}")
        print("Please run the main simulation pipeline first (pipeline.py)")
        return False
    return True


def load_data(data_dir: str = '.') -> Dict[str, Any]:
    """Load all available data files from the specified directory."""
    print("Loading data files...")

    data = {}

    # Load time vector (required)
    time_path = os.path.join(data_dir, config.TIME_VECTOR_CSV)
    data['time_test'] = np.loadtxt(time_path, delimiter=',')
    print(f"  ✓ Loaded {config.TIME_VECTOR_CSV}: {len(data['time_test'])} time points")

    # Load parameter names (required)
    param_path = os.path.join(data_dir, config.PARAM_NAMES_TXT)
    with open(param_path, 'r') as f:
        data['param_names'] = [line.strip() for line in f]
    print(f"  ✓ Loaded {config.PARAM_NAMES_TXT}: {len(data['param_names'])} parameters")

    # Load mean voltage (required)
    mean_path = os.path.join(data_dir, config.MEAN_V_CSV)
    data['mean_V'] = np.loadtxt(mean_path, delimiter=',')
    print(f"  ✓ Loaded {config.MEAN_V_CSV}")

    # Load standard deviation (optional)
    std_path = os.path.join(data_dir, config.STD_V_CSV)
    if os.path.exists(std_path):
        data['std_V'] = np.loadtxt(std_path, delimiter=',')
        print(f"  ✓ Loaded {config.STD_V_CSV}")

    # Load sample traces (optional)
    sample_path = os.path.join(data_dir, config.ALL_V_SAMPLE_CSV)
    if os.path.exists(sample_path):
        data['all_V_sample'] = np.loadtxt(sample_path, delimiter=',')
        print(f"  ✓ Loaded {config.ALL_V_SAMPLE_CSV}: {data['all_V_sample'].shape[0]} traces")

    # Load Sobol indices (with or without headers)
    s1_with_headers = os.path.join(data_dir, config.S1_TIME_WITH_HEADERS_CSV)
    s1_plain = os.path.join(data_dir, config.S1_TIME_CSV)

    if os.path.exists(s1_with_headers):
        S1_df = pd.read_csv(s1_with_headers)
        data['S1_time'] = S1_df.values
        print(f"  ✓ Loaded {config.S1_TIME_WITH_HEADERS_CSV}")
    elif os.path.exists(s1_plain):
        data['S1_time'] = np.loadtxt(s1_plain, delimiter=',')
        print(f"  ✓ Loaded {config.S1_TIME_CSV}")

    st_with_headers = os.path.join(data_dir, config.ST_TIME_WITH_HEADERS_CSV)
    st_plain = os.path.join(data_dir, config.ST_TIME_CSV)

    if os.path.exists(st_with_headers):
        ST_df = pd.read_csv(st_with_headers)
        data['ST_time'] = ST_df.values
        print(f"  ✓ Loaded {config.ST_TIME_WITH_HEADERS_CSV}")
    elif os.path.exists(st_plain):
        data['ST_time'] = np.loadtxt(st_plain, delimiter=',')
        print(f"  ✓ Loaded {config.ST_TIME_CSV}")

    print("\nData loading complete!\n")
    return data


def main():
    """Main function to regenerate all figures."""
    import argparse

    parser = argparse.ArgumentParser(description="Replot HH Sensitivity Analysis Figures")
    parser.add_argument('--data-dir', default='.',
                       help='Directory containing the data files (default: current directory)')
    parser.add_argument('--output-dir', default=None,
                       help='Directory to save figures (default: same as data-dir)')
    parser.add_argument('--skip-traces', action='store_true',
                       help='Skip voltage trace plots')
    parser.add_argument('--skip-sobol-time', action='store_true',
                       help='Skip time-dependent Sobol plots')
    parser.add_argument('--skip-sobol-bars', action='store_true',
                       help='Skip bar chart Sobol plots')
    parser.add_argument('--skip-combined', action='store_true',
                       help='Skip combined A4 layout')

    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir if args.output_dir else data_dir

    print("="*80)
    print("REGENERATING PUBLICATION-QUALITY FIGURES")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Check if required files exist
    if not check_required_files(data_dir):
        sys.exit(1)

    # Load all data
    data = load_data(data_dir)

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate plots based on available data and user preferences
    print("Generating figures...")
    print()

    generated_plots = []

    # Voltage trace plots (if sample data available and not skipped)
    if not args.skip_traces and 'all_V_sample' in data:
        plotting.plot_voltage_traces(data, output_dir)
        plotting.plot_voltage_traces_with_stimulus(data, output_dir)
        generated_plots.extend(['voltage_traces', 'voltage_traces_with_stimulus'])

    # Sobol time series plots (if data available and not skipped)
    if not args.skip_sobol_time:
        if 'S1_time' in data:
            plotting.plot_sobol_S1(data, output_dir)
            generated_plots.append('sobol_S1')

        if 'ST_time' in data:
            plotting.plot_sobol_ST(data, output_dir)
            generated_plots.append('sobol_ST')

    # Sobol bar plots (if data available and not skipped)
    if not args.skip_sobol_bars:
        if 'S1_time' in data:
            plotting.plot_sobol_S1_at_70ms(data, output_dir)
            generated_plots.append('sobol_S1_70ms')

        if 'ST_time' in data:
            plotting.plot_sobol_ST_at_70ms(data, output_dir)
            generated_plots.append('sobol_ST_70ms')

    # Combined A4 layout (if data available and not skipped)
    if not args.skip_combined and 'S1_time' in data and 'ST_time' in data:
        plotting.plot_combined_A4_layout(data, output_dir)
        generated_plots.append('sobol_combined_A4')

    print()
    print("="*80)
    print("FIGURE REGENERATION COMPLETE!")
    print("="*80)
    print()
    print("Generated figures (PNG + PDF):")

    if 'voltage_traces' in generated_plots:
        print("  ✓ voltage_traces.png/.pdf")
        print("  ✓ voltage_traces_with_stimulus.png/.pdf")

    if 'sobol_S1' in generated_plots:
        print("  ✓ sobol_S1.png/.pdf")

    if 'sobol_ST' in generated_plots:
        print("  ✓ sobol_ST.png/.pdf")

    if 'sobol_S1_70ms' in generated_plots:
        print("  ✓ sobol_S1_70ms.png/.pdf")

    if 'sobol_ST_70ms' in generated_plots:
        print("  ✓ sobol_ST_70ms.png/.pdf")

    if 'sobol_combined_A4' in generated_plots:
        print("  ✓ sobol_combined_A4.png/.pdf")

    print()
    print(f"All figures saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()