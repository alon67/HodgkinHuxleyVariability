"""
Main module for the Hodgkin-Huxley cable model Monte Carlo simulation and analysis.

This module provides a command-line interface to run the complete workflow:
1. Load/generate parameter sets
2. Run JAX-accelerated simulations
3. Perform comprehensive analysis
4. Generate publication-quality figures

Usage:
    python main.py --n-samples 10000 --save-path ./results
"""

import argparse
import os
import sys
import numpy as np
# Monkey patch for JAX 0.8.1 compatibility with numpy 1.26.4
original_asarray = np.asarray
def patched_asarray(a, dtype=None, order=None, copy=None):
    try:
        if copy is not None:
            return original_asarray(a, dtype=dtype, order=order, copy=copy)
        else:
            return original_asarray(a, dtype=dtype, order=order)
    except TypeError as e:
        if 'copy' in str(e):
            # Fallback for numpy versions that don't support copy
            return original_asarray(a, dtype=dtype, order=order)
        else:
            raise
np.asarray = patched_asarray

import jax
import jax.numpy as jnp
from datetime import datetime

# Import our modular components
from config import (
    print_configuration, configure_xla_environment,
    DEFAULT_CHUNK_SIZE, DEFAULT_SAVE_PATH, DEFAULT_N_SAMPLES,
    DEFAULT_I_AMPLITUDE, T_START, T_END, DEFAULT_BOOTSTRAP_FILE,
    DEFAULT_PARAM_FILE, DT, N_SEGMENTS
)
from parameters import (
    load_bootstrap_stats, load_or_generate_param_values,
    param_names, get_latex_name
)
from core_simulation import (
    simulate_batch_jax, incremental_simulation_jax,
    find_equilibrium_jax, simulate_HH_AP_jax
)
from core_analysis import (
    analyze_resting_potentials, categorize_traces,
    analyze_firing_frequency, analyze_stable_potential_after_spike,
    analyze_ap_propagation, compute_statistics_from_disk,
    save_sample_traces
)
from graphics import (
    plot_pie_chart, plot_representative_traces,
    plot_parameter_scatter_matrix, plot_parameter_distributions_by_category,
    plot_firing_frequency_histogram, plot_stable_potential_histogram,
    plot_ap_amplitude_histograms, plot_propagation_speed_histogram,
    plot_single_ap_traces_overlay, plot_mean_trace_with_std
)


def run_simulations(n_samples, save_path, param_values):
    """
    Run the JAX-accelerated Monte Carlo simulations.

    Parameters:
    -----------
    n_samples : int
        Number of parameter sets to simulate
    save_path : str
        Directory to save simulation results
    param_values : array
        Parameter sets to simulate

    Returns:
    --------
    num_chunks : int
        Number of chunks created
    time_test : array
        Time vector used in simulations
    """
    print("\n" + "="*80)
    print("STARTING JAX-ACCELERATED MONTE CARLO SIMULATIONS")
    print("="*80)
    print(f"Number of samples: {n_samples}")
    print(f"Save path: {save_path}")
    print(f"JAX backend: {os.environ.get('JAX_PLATFORM_NAME', 'cpu')}")

    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Configure JAX environment
    configure_xla_environment()

    # Compute base parameters (means from bootstrap data)
    bootstrap_stats = load_bootstrap_stats(DEFAULT_BOOTSTRAP_FILE)
    base_params = [mean for mean, std in bootstrap_stats]
    
    # Compute equilibrium-based initial conditions
    print("\nComputing equilibrium initial conditions...")
    base_params_jax = jnp.array(base_params)
    V_eq_arr, m_eq_arr, h_eq_arr, n_eq_arr = find_equilibrium_jax(base_params_jax)
    print(".3f")
    
    # Run test simulation to get time vector
    print("Running test simulation to determine time vector...")
    time_test, V_inj_test, V_dist_test = simulate_HH_AP_jax(
        base_params_jax, 
        (V_eq_arr, m_eq_arr, h_eq_arr, n_eq_arr),
        tmax=100.0, 
        dt=DT,
        I_amplitude=DEFAULT_I_AMPLITUDE
    )
    time_test = np.array(time_test)  # Convert to NumPy
    dt = time_test[1] - time_test[0]
    nT = len(time_test)
    print(f"Test simulation complete. Time points: {nT}, dt: {dt:.6f} ms")

    # Run simulations using incremental approach for memory efficiency
    print("\nRunning incremental simulations...")
    save_path_returned, num_chunks = incremental_simulation_jax(
        param_values, time_test, dt, nT, 
        chunk_size=DEFAULT_CHUNK_SIZE, 
        save_path=save_path,
        I_amplitude=DEFAULT_I_AMPLITUDE
    )

    print(f"\nSimulation complete! Created {num_chunks} chunks with {n_samples} total simulations")
    print(f"Time vector length: {len(time_test)} points")
    print(f"Simulation time range: {time_test[0]:.1f} to {time_test[-1]:.1f} ms")

    return num_chunks, time_test


def run_analysis(save_path, num_chunks, time_test, param_values, param_bounds):
    """
    Run the complete analysis pipeline.

    Parameters:
    -----------
    save_path : str
        Directory containing simulation results
    num_chunks : int
        Number of chunks to analyze
    time_test : array
        Time vector
    param_values : array
        Parameter sets used in simulations
    """
    print("\n" + "="*80)
    print("STARTING COMPREHENSIVE ANALYSIS")
    print("="*80)

    # 1. Analyze resting membrane potentials
    print("\n1. Analyzing resting membrane potentials...")
    resting_potentials = analyze_resting_potentials(save_path, num_chunks)

    # 2. Categorize traces based on action potential patterns
    print("\n2. Categorizing voltage traces...")
    categories = categorize_traces(save_path, num_chunks, time_test, param_values=param_values)

    # 3. Create visualizations
    print("\n3. Creating publication-quality figures...")

    # Pie chart of trace categories
    plot_pie_chart(categories)

    # Representative traces for each category
    plot_representative_traces(categories, time_test)

    # Parameter distributions by category
    plot_parameter_distributions_by_category(categories, param_values, param_names, param_bounds)

    # Parameter scatter matrix for propagated category
    plot_parameter_scatter_matrix(categories, param_values, param_names, param_bounds)

    # 4. Analyze firing patterns
    print("\n4. Analyzing firing patterns...")

    # Firing frequency analysis
    firing_frequencies = analyze_firing_frequency(save_path, num_chunks, time_test)
    if len(firing_frequencies) > 0:
        plot_firing_frequency_histogram(firing_frequencies)

    # Stable potential after single spike
    stable_potentials = analyze_stable_potential_after_spike(save_path, num_chunks, time_test,
                                                             measurement_time=70.0)
    if len(stable_potentials) > 0:
        plot_stable_potential_histogram(stable_potentials, measurement_time=70.0)

    # 5. Analyze action potential propagation
    print("\n5. Analyzing action potential propagation...")
    propagation_results = analyze_ap_propagation(save_path, num_chunks, time_test)

    # Create propagation plots
    if propagation_results:
        plot_ap_amplitude_histograms(propagation_results)
        if len(propagation_results['speeds']) > 0:
            plot_propagation_speed_histogram(propagation_results['speeds'])

        # Plot representative single AP traces
        if 'all_single_ap_traces_inj' in propagation_results:
            # This would need to be modified to store traces in the results
            pass

    # 6. Compute and plot mean trace with standard deviation
    print("\n6. Computing mean trace with standard deviation...")
    try:
        mean_V, std_V = compute_statistics_from_disk(save_path, num_chunks, time_test)
        plot_mean_trace_with_std(time_test, mean_V, std_V)
    except Exception as e:
        print(f"Warning: Could not compute mean trace statistics: {e}")

    # 7. Save sample traces
    print("\n7. Saving sample traces...")
    try:
        save_sample_traces(save_path, num_chunks, num_samples=1000)
    except Exception as e:
        print(f"Warning: Could not save sample traces: {e}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)


def cleanup_simulation_data(save_path, num_chunks):
    """
    Clean up temporary simulation data files if requested.

    Parameters:
    -----------
    save_path : str
        Directory containing simulation data
    num_chunks : int
        Number of chunks to clean up
    """
    if not False:
        print("\nCleaning up temporary simulation files...")
        files_removed = 0

        # Remove chunk files
        for chunk_idx in range(num_chunks):
            # Remove main chunk files
            for suffix in ['_injection.npy', '_distal.npy']:
                chunk_file = os.path.join(save_path, f'chunk_{chunk_idx}{suffix}')
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
                    files_removed += 1

            # Remove sub-chunk files
            sub_idx = 0
            while True:
                sub_file_exists = False
                for suffix in ['_injection.npy', '_distal.npy']:
                    sub_file = os.path.join(save_path, f'chunk_{chunk_idx}_sub_{sub_idx}{suffix}')
                    if os.path.exists(sub_file):
                        os.remove(sub_file)
                        files_removed += 1
                        sub_file_exists = True
                if not sub_file_exists:
                    break
                sub_idx += 1

        # Remove directory if empty
        try:
            os.rmdir(save_path)
            print(f"Removed {files_removed} temporary files and cleaned up directory")
        except OSError:
            print(f"Removed {files_removed} temporary files (directory not empty)")
    else:
        print(f"\nKeeping simulation data in: {save_path}")


def save_summary_report(categories, propagation_results=None):
    """
    Save a comprehensive summary report of the analysis.

    Parameters:
    -----------
    categories : dict
        Trace categorization results
    propagation_results : dict, optional
        Propagation analysis results
    """
    print("\nSaving comprehensive summary report...")

    try:
        with open('analysis_summary_report.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write("HODGKIN-HUXLEY CABLE MODEL - MONTE CARLO ANALYSIS SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Stimulus amplitude: {DEFAULT_I_AMPLITUDE} μA\n")
            f.write(f"Stimulus duration: {T_START}-{T_END} ms\n\n")

            # Trace categorization
            total_traces = sum(cat['count'] for cat in categories.values())
            f.write("TRACE CATEGORIZATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total traces analyzed: {total_traces:,}\n\n")

            category_descriptions = {
                'no_ap': 'No Action Potential',
                'single_ap_propagated': 'Single AP - Propagated',
                'single_ap_failed': 'Single AP - Failed Propagation',
                'multiple_ap_during_stim': 'Multiple APs During Stimulus',
                'ap_before_stim': 'AP Before Stimulus'
            }

            for category, data in categories.items():
                percentage = (data['count'] / total_traces) * 100 if total_traces > 0 else 0
                f.write(f"{category_descriptions.get(category, category)}: {data['count']:,} ({percentage:.1f}%)\n")
            f.write("\n")

            # Propagation analysis
            if propagation_results and 'summary' in propagation_results:
                summary = propagation_results['summary']
                f.write("PROPAGATION ANALYSIS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Traces with single AP: {summary['single_ap_traces']:,}\n")
                f.write(f"Successful propagation: {summary['successful_propagation']:,} ({summary['success_rate']:.1f}%)\n")
                f.write(f"Failed propagation: {summary['failed_propagation']:,} ({summary['failure_rate']:.1f}%)\n")

                if len(propagation_results.get('speeds', [])) > 0:
                    speeds = propagation_results['speeds']
                    f.write(f"\nPropagation speed: {np.mean(speeds):.2f} ± {np.std(speeds):.2f} m/s (n={len(speeds)})\n")

            f.write("\n" + "="*80 + "\n")

        print("✓ Saved analysis_summary_report.txt")

    except Exception as e:
        print(f"Warning: Could not save summary report: {e}")


def main():
    """
    Main entry point for the Hodgkin-Huxley cable model analysis.
    """
    parser = argparse.ArgumentParser(
        description='JAX-accelerated Hodgkin-Huxley cable model Monte Carlo analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run analysis with 10,000 samples
  python main.py --n-samples 10000

  # Run with custom save path
  python main.py --n-samples 5000 --save-path ./my_results

  # Skip simulation and analyze existing data
  python main.py --analyze-only --save-path ./existing_results
        """
    )

    parser.add_argument('--n-samples', type=int, default=1000,
                        help='Number of Monte Carlo samples (default: 1000)')
    parser.add_argument('--save-path', type=str, default=DEFAULT_SAVE_PATH,
                        help=f'Directory to save results (default: {DEFAULT_SAVE_PATH})')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Skip simulation and analyze existing data')
    parser.add_argument('--config-only', action='store_true',
                        help='Print configuration and exit')

    args = parser.parse_args()

    # Print configuration
    print_configuration()

    if args.config_only:
        return

    # Load bootstrap statistics and generate/load parameter values
    print("\nLoading parameter statistics...")
    bootstrap_stats = load_bootstrap_stats(DEFAULT_BOOTSTRAP_FILE)
    param_values = load_or_generate_param_values(DEFAULT_BOOTSTRAP_FILE, DEFAULT_PARAM_FILE, args.n_samples)
    param_bounds = [(param_values[:, i].min(), param_values[:, i].max()) 
                    for i in range(len(param_names))]

    print(f"Parameter sets: {param_values.shape[0]} samples, {param_values.shape[1]} parameters")

    # Run simulations unless analyze-only is specified
    if not args.analyze_only:
        num_chunks, time_test = run_simulations(args.n_samples, args.save_path, param_values)
    else:
        # For analyze-only, we need to determine num_chunks and time_test from existing data
        print(f"\nAnalyzing existing data in: {args.save_path}")

        # Count chunks
        num_chunks = 0
        while True:
            chunk_file = os.path.join(args.save_path, f'chunk_{num_chunks}_injection.npy')
            if os.path.exists(chunk_file):
                num_chunks += 1
            else:
                break

        if num_chunks == 0:
            print("ERROR: No simulation data found in the specified directory")
            sys.exit(1)

        # Load time vector (assume it's saved somewhere, or reconstruct)
        # For now, we'll use a default time vector - this should be improved
        from config import T_START, T_END, DT
        time_test = np.arange(T_START, T_END + DT, DT)

        print(f"Found {num_chunks} chunks of existing simulation data")

    # Run analysis
    run_analysis(args.save_path, num_chunks, time_test, param_values, param_bounds)

    # Save summary report
    # Note: We'd need to capture the results from run_analysis to pass here
    # For now, just create a basic report
    save_summary_report({})  # Empty dict for now

    # Cleanup
    if not args.analyze_only:
        cleanup_simulation_data(args.save_path, num_chunks)

    print("\n" + "="*80)
    print("WORKFLOW COMPLETE!")
    print("="*80)
    print("\nAll results saved to the current directory.")
    print("Check the generated figures and data files for detailed analysis.")


if __name__ == '__main__':
    main()