# pipeline.py
"""
End-to-end pipeline for Hodgkin-Huxley sensitivity analysis.
Orchestrates the complete workflow from parameter sampling to figure generation.
"""

import numpy as np
import pandas as pd
import os
import sys
import gc
import time
from multiprocessing import cpu_count
from typing import Dict, Any, Optional
import config
import parameters
import sampling
import sim_core
import sim_io
import analysis
import plotting


def run_pipeline(bootstrap_file: str = config.PARAMS_BOOT_CSV,
                N: int = config.N_DEFAULT,
                chunk_size: int = config.CHUNK_SIZE_DEFAULT,
                save_path: str = config.SAVE_PATH_DEFAULT,
                filtered_save_path: str = config.FILTERED_SAVE_PATH_DEFAULT,
                tmax: float = config.TMAX,
                dt: float = config.DT,
                I_amplitude: float = config.I_AMPLITUDE_DEFAULT,
                skip_simulation: bool = False,
                skip_filtering: bool = False) -> None:
    """
    Run the complete HH sensitivity analysis pipeline.

    Parameters
    ----------
    bootstrap_file : str
        Path to params_boot.csv
    N : int
        Base sample size for Saltelli sampling
    chunk_size : int
        Simulations per chunk
    save_path : str
        Directory for simulation chunks
    filtered_save_path : str
        Directory for filtered chunks
    tmax : float
        Maximum simulation time
    dt : float
        Time step
    I_amplitude : float
        Current injection amplitude
    skip_simulation : bool
        Skip simulation if chunks exist
    skip_filtering : bool
        Skip filtering if masked data exists
    """
    print("="*80)
    print("HODGKIN-HUXLEY SENSITIVITY ANALYSIS PIPELINE")
    print("="*80)
    print()

    # Validate configuration
    config.validate_config()
    config.print_config()

    #--------------------------------------------------------------------------
    # 1) Load bootstrap parameters and build problem
    #--------------------------------------------------------------------------
    print("\n" + "="*60)
    print("LOADING PARAMETERS AND BUILDING PROBLEM")
    print("="*60)

    problem, base_params, bounds, param_names = sampling.build_problem_from_bootstrap(bootstrap_file)
    print(f"✓ Loaded {len(param_names)} parameters from {bootstrap_file}")
    print(f"✓ Parameter bounds: {len(bounds)} ranges defined")

    #--------------------------------------------------------------------------
    # 2) Generate Saltelli samples
    #--------------------------------------------------------------------------
    print("\n" + "="*60)
    print("GENERATING SALIELLI SAMPLES")
    print("="*60)

    param_values = sampling.generate_saltelli_samples(problem, N, calc_second_order=config.CALC_SECOND_ORDER)
    sampling.validate_sample_structure(param_values, problem, config.CALC_SECOND_ORDER)

    #--------------------------------------------------------------------------
    # 3) Compute equilibrium initial conditions
    #--------------------------------------------------------------------------
    print("\n" + "="*60)
    print("COMPUTING EQUILIBRIUM INITIAL CONDITIONS")
    print("="*60)

    base_params_jax = sim_core.jnp.array(base_params)
    V_eq, m_eq, h_eq, n_eq = sim_core.find_equilibrium_jax(base_params_jax)
    V_eq_val = float(V_eq)
    m_eq_val = float(m_eq)
    h_eq_val = float(h_eq)
    n_eq_val = float(n_eq)
    print(f"✓ Calculated V_rest = {V_eq_val:.3f} mV")
    print(f"  (m0={m_eq_val:.3f}, h0={h_eq_val:.3f}, n0={n_eq_val:.3f})")

    #--------------------------------------------------------------------------
    # 4) Test simulation to define time vector
    #--------------------------------------------------------------------------
    print("\n" + "="*60)
    print("TEST SIMULATION FOR TIME VECTOR")
    print("="*60)

    time_test, test_sol = sim_core.simulate_HH_AP_jax(
        base_params_jax,
        (V_eq, m_eq, h_eq, n_eq),
        tmax=tmax,
        dt=dt,
        I_amplitude=I_amplitude
    )
    time_test = np.array(time_test)
    dt_actual = time_test[1] - time_test[0]
    nT = len(time_test)

    print(f"✓ Test simulation completed")
    print(f"  Time points: {nT}, dt={dt_actual:.6f} ms")
    print(f"  Total samples: {len(param_values)}")

    # Save time vector
    np.savetxt(config.TIME_VECTOR_CSV, time_test, delimiter=',')
    print(f"✓ Saved time vector to {config.TIME_VECTOR_CSV}")

    #--------------------------------------------------------------------------
    # 5) Check for existing simulation data
    #--------------------------------------------------------------------------
    if skip_simulation:
        print("\n" + "="*60)
        print("CHECKING FOR EXISTING SIMULATION DATA")
        print("="*60)

        if os.path.exists(save_path):
            num_chunks_existing = sim_io.infer_num_chunks(save_path)
            expected_chunks = int(np.ceil(len(param_values) / chunk_size))
            if num_chunks_existing >= expected_chunks:
                print(f"✓ Found {num_chunks_existing} existing chunks (expected {expected_chunks})")
                skip_simulation = True
            else:
                print(f"⚠ Only {num_chunks_existing}/{expected_chunks} chunks found - will run simulation")
                skip_simulation = False
        else:
            print("⚠ No existing simulation data found - will run simulation")
            skip_simulation = False

    #--------------------------------------------------------------------------
    # 6) Run incremental simulations
    #--------------------------------------------------------------------------
    if not skip_simulation:
        print("\n" + "="*60)
        print("RUNNING JAX SIMULATIONS")
        print("="*60)

        save_path_actual, num_chunks = sim_io.incremental_simulation_jax(
            param_values, time_test, dt_actual, nT,
            chunk_size=chunk_size, save_path=save_path, I_amplitude=I_amplitude
        )
        save_path = save_path_actual
    else:
        print("\n" + "="*60)
        print("SKIPPING SIMULATION - USING EXISTING DATA")
        print("="*60)
        num_chunks = sim_io.infer_num_chunks(save_path)

    # JAX cleanup
    sim_core.jax.clear_caches()
    gc.collect()

    #--------------------------------------------------------------------------
    # 7) Categorize traces and create masked data
    #--------------------------------------------------------------------------
    if not skip_filtering:
        print("\n" + "="*60)
        print("FILTERING DATA FOR SINGLE AP TRACES")
        print("="*60)

        single_ap_indices, categories_count = analysis.categorize_traces(
            save_path, num_chunks, time_test,
            stimulus_start=config.STIMULUS_START, stimulus_end=config.STIMULUS_END,
            ap_threshold=config.AP_THRESHOLD
        )

        if len(single_ap_indices) == 0:
            print("\nERROR: No single AP traces found! Cannot proceed with sensitivity analysis.")
            print("Consider adjusting the ap_threshold or checking the simulation parameters.")
            sys.exit(1)

        filtered_save_path_actual, num_filtered_chunks = analysis.create_masked_data(
            save_path, num_chunks, single_ap_indices, nT, filtered_save_path
        )

        if filtered_save_path_actual is None:
            print("\nERROR: Failed to create masked data!")
            sys.exit(1)

        filtered_save_path = filtered_save_path_actual
    else:
        print("\n" + "="*60)
        print("SKIPPING FILTERING - USING EXISTING MASKED DATA")
        print("="*60)
        num_filtered_chunks = sim_io.infer_num_chunks(filtered_save_path)

    # Update paths for analysis
    original_save_path = save_path
    original_num_chunks = num_chunks
    save_path = filtered_save_path
    num_chunks = num_filtered_chunks

    print(f"\n✓ Using {num_chunks} filtered chunks for sensitivity analysis")
    print(f"  Original path: {original_save_path} ({original_num_chunks} chunks)")
    print(f"  Masked path: {save_path} ({num_chunks} chunks)")

    #--------------------------------------------------------------------------
    # 8) Transpose data for Sobol analysis
    #--------------------------------------------------------------------------
    print("\n" + "="*60)
    print("TRANSPOSING DATA FOR SOBOL ANALYSIS")
    print("="*60)

    sim_io.transpose_chunks_for_sobol(save_path, num_chunks, nT)
    save_path = filtered_save_path
    num_chunks = num_filtered_chunks

    # Transpose masked data
    sim_io.transpose_chunks_for_sobol(save_path, num_chunks, nT)

    #--------------------------------------------------------------------------
    # 9) Run Sobol analysis
    #--------------------------------------------------------------------------
    print("\n" + "="*60)
    print("RUNNING SOBOL SENSITIVITY ANALYSIS")
    print("="*60)

    # Determine optimal processes
    if N >= 16384:
        num_sobol_processes = 8
    else:
        num_sobol_processes = max(4, cpu_count() - 4)

    print(f"Using {num_sobol_processes} processes for Sobol analysis")

    S1_time, ST_time = analysis.batch_sobol_analysis_from_disk(
        save_path, num_chunks, problem, num_sobol_processes, batch_size=500
    )

    sobol_success = np.sum(~np.all(S1_time == 0, axis=1))
    print(f"✓ Sobol analysis completed: {sobol_success}/{nT} time points successful")

    #--------------------------------------------------------------------------
    # 10) Compute statistics
    #--------------------------------------------------------------------------
    print("\n" + "="*60)
    print("COMPUTING STATISTICS")
    print("="*60)

    mean_V, std_V = analysis.compute_statistics_from_disk(save_path, num_chunks, time_test)

    #--------------------------------------------------------------------------
    # 11) Save sample traces
    #--------------------------------------------------------------------------
    print("\n" + "="*60)
    print("SAVING SAMPLE TRACES")
    print("="*60)

    analysis.save_sample_traces(save_path, num_chunks, config.ALL_V_SAMPLE_CSV, num_samples=1000)

    # Save transposed sample data
    try:
        all_V_sample = np.loadtxt(config.ALL_V_SAMPLE_CSV, delimiter=',')
        all_V_sample_transposed = all_V_sample.T
        np.savetxt(config.ALL_V_SAMPLE_TRANSPOSED_CSV, all_V_sample_transposed, delimiter=',', fmt='%.6f')
        print(f"✓ Saved transposed sample data ({all_V_sample_transposed.shape[0]} time points × {all_V_sample_transposed.shape[1]} samples)")
    except Exception as e:
        print(f"⚠ Could not create transposed sample data: {e}")

    #--------------------------------------------------------------------------
    # 12) Save results
    #--------------------------------------------------------------------------
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    # Save statistics
    np.savetxt(config.MEAN_V_CSV, mean_V, delimiter=',')
    np.savetxt(config.STD_V_CSV, std_V, delimiter=',')

    # Save Sobol indices
    np.savetxt(config.S1_TIME_CSV, S1_time, delimiter=',')
    np.savetxt(config.ST_TIME_CSV, ST_time, delimiter=',')

    # Save parameter names
    parameters.save_param_names_txt(param_names, config.PARAM_NAMES_TXT)

    # Add parameter headers to Sobol index CSV files
    try:
        S1_df = pd.DataFrame(S1_time, columns=param_names)
        ST_df = pd.DataFrame(ST_time, columns=param_names)

        S1_df.to_csv(config.S1_TIME_WITH_HEADERS_CSV, index=False)
        ST_df.to_csv(config.ST_TIME_WITH_HEADERS_CSV, index=False)
        print("✓ Saved Sobol indices with parameter headers")
    except Exception as e:
        print(f"⚠ Could not add parameter headers: {e}")

    #--------------------------------------------------------------------------
    # 13) Generate figures
    #--------------------------------------------------------------------------
    print("\n" + "="*60)
    print("GENERATING FIGURES")
    print("="*60)

    # Prepare data dict for plotting
    plot_data = {
        'time_test': time_test,
        'param_names': param_names,
        'mean_V': mean_V,
        'std_V': std_V,
        'S1_time': S1_time,
        'ST_time': ST_time,
        'all_V_sample': all_V_sample if 'all_V_sample' in locals() else None
    }

    # Generate all plots
    plotting.plot_voltage_traces(plot_data)
    plotting.plot_voltage_traces_with_stimulus(plot_data)
    plotting.plot_sobol_S1(plot_data)
    plotting.plot_sobol_ST(plot_data)
    plotting.plot_sobol_S1_at_70ms(plot_data)
    plotting.plot_sobol_ST_at_70ms(plot_data)
    plotting.plot_combined_A4_layout(plot_data)

    #--------------------------------------------------------------------------
    # 14) Clean up temporary files
    #--------------------------------------------------------------------------
    print("\n" + "="*60)
    print("CLEANING UP TEMPORARY FILES")
    print("="*60)

    files_removed = 0
    for chunk_idx in range(num_chunks):
        # Remove main chunk file
        chunk_file = os.path.join(save_path, f'chunk_{chunk_idx}.npy')
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
            files_removed += 1

        # Remove transposed file
        trans_file = os.path.join(save_path, f'chunk_{chunk_idx}_transposed.npy')
        if os.path.exists(trans_file):
            os.remove(trans_file)
            files_removed += 1

        # Remove sub-chunk files
        sub_idx = 0
        while True:
            sub_chunk_file = os.path.join(save_path, f'chunk_{chunk_idx}_sub_{sub_idx}.npy')
            if os.path.exists(sub_chunk_file):
                os.remove(sub_chunk_file)
                files_removed += 1
                sub_idx += 1
            else:
                break

            # Also check for transposed sub-chunks
            sub_trans_file = os.path.join(save_path, f'chunk_{chunk_idx}_sub_{sub_idx}_transposed.npy')
            if os.path.exists(sub_trans_file):
                os.remove(sub_trans_file)
                files_removed += 1

    # Remove directory if it's empty
    try:
        os.rmdir(save_path)
        print(f"✓ Removed {files_removed} temporary files and cleaned up directory")
    except OSError:
        print(f"✓ Removed {files_removed} temporary files (directory not empty)")

    #--------------------------------------------------------------------------
    # 15) Final summary
    #--------------------------------------------------------------------------
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Analysis method: Non-single-AP traces replaced with mean of single-AP traces")
    print(f"This preserves Saltelli sample structure for valid Sobol analysis")
    print()
    print("Generated files:")
    print(f"  - Statistics: {config.MEAN_V_CSV}, {config.STD_V_CSV}")
    print(f"  - Sobol indices: {config.S1_TIME_CSV}, {config.ST_TIME_CSV}")
    print(f"  - With headers: {config.S1_TIME_WITH_HEADERS_CSV}, {config.ST_TIME_WITH_HEADERS_CSV}")
    print(f"  - Sample data: {config.ALL_V_SAMPLE_CSV}, {config.ALL_V_SAMPLE_TRANSPOSED_CSV}")
    print(f"  - Metadata: {config.TIME_VECTOR_CSV}, {config.PARAM_NAMES_TXT}")
    print()
    print("Generated figures (PNG + PDF):")
    print("  - voltage_traces.png/.pdf")
    print("  - voltage_traces_with_stimulus.png/.pdf")
    print("  - sobol_S1.png/.pdf, sobol_ST.png/.pdf")
    print("  - sobol_S1_70ms.png/.pdf, sobol_ST_70ms.png/.pdf")
    print("  - sobol_combined_A4.png/.pdf")
    print()
    print(f"Note: Original simulation data preserved in '{original_save_path}'")
    print(f"      Masked data directory '{save_path}' cleaned up")
    print("="*80)


if __name__ == "__main__":
    # Simple CLI - could be enhanced with argparse
    import argparse

    parser = argparse.ArgumentParser(description="HH Sensitivity Analysis Pipeline")
    parser.add_argument('--bootstrap-file', default=config.PARAMS_BOOT_CSV,
                       help='Path to params_boot.csv')
    parser.add_argument('--N', type=int, default=config.N_DEFAULT,
                       help='Base sample size for Saltelli sampling')
    parser.add_argument('--chunk-size', type=int, default=config.CHUNK_SIZE_DEFAULT,
                       help='Simulations per chunk')
    parser.add_argument('--save-path', default=config.SAVE_PATH_DEFAULT,
                       help='Directory for simulation chunks')
    parser.add_argument('--filtered-save-path', default=config.FILTERED_SAVE_PATH_DEFAULT,
                       help='Directory for filtered chunks')
    parser.add_argument('--tmax', type=float, default=config.TMAX,
                       help='Maximum simulation time')
    parser.add_argument('--dt', type=float, default=config.DT,
                       help='Time step')
    parser.add_argument('--I-amplitude', type=float, default=config.I_AMPLITUDE_DEFAULT,
                       help='Current injection amplitude')
    parser.add_argument('--skip-simulation', action='store_true',
                       help='Skip simulation if chunks exist')
    parser.add_argument('--skip-filtering', action='store_true',
                       help='Skip filtering if masked data exists')

    args = parser.parse_args()

    run_pipeline(
        bootstrap_file=args.bootstrap_file,
        N=args.N,
        chunk_size=args.chunk_size,
        save_path=args.save_path,
        filtered_save_path=args.filtered_save_path,
        tmax=args.tmax,
        dt=args.dt,
        I_amplitude=args.I_amplitude,
        skip_simulation=args.skip_simulation,
        skip_filtering=args.skip_filtering
    )