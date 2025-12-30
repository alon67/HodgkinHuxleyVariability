# analysis.py
"""
Analysis functions for trace categorization, masking, and Sobol computation.
Handles all post-simulation data processing.
"""

import numpy as np
import os
import glob
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from typing import List, Tuple, Dict, Any, Optional
import config


def categorize_traces(save_path: str, num_chunks: int, time_test: np.ndarray,
                     stimulus_start: float = config.STIMULUS_START,
                     stimulus_end: float = config.STIMULUS_END,
                     ap_threshold: float = config.AP_THRESHOLD) -> Tuple[List[Tuple[int, int]], Dict[str, int]]:
    """
    Categorize voltage traces based on action potential patterns and return indices.
    Focus on identifying single AP after stimulus onset traces for sensitivity analysis.

    Parameters
    ----------
    save_path : str
        Directory containing chunk files
    num_chunks : int
        Number of chunk files
    time_test : np.ndarray
        Time vector
    stimulus_start : float
        Start time of stimulus (ms)
    stimulus_end : float
        End time of stimulus (ms)
    ap_threshold : float
        Voltage threshold for detecting action potentials (mV)

    Returns
    -------
    single_ap_indices : List[Tuple[int, int]]
        List of tuples (chunk_file_idx, trace_idx) for traces with single AP after stimulus
    categories_count : Dict[str, int]
        Dictionary with counts for each category
    """
    print("Categorizing voltage traces to identify single AP patterns...")

    # Find all chunk files
    chunk_files = []
    chunk_indices = []
    for chunk_idx in range(num_chunks):
        main_chunk_file = os.path.join(save_path, f'chunk_{chunk_idx}.npy')
        if os.path.exists(main_chunk_file):
            chunk_files.append(main_chunk_file)
            chunk_indices.append(chunk_idx)
        else:
            # Look for sub-chunks
            sub_idx = 0
            while True:
                sub_chunk_file = os.path.join(save_path, f'chunk_{chunk_idx}_sub_{sub_idx}.npy')
                if os.path.exists(sub_chunk_file):
                    chunk_files.append(sub_chunk_file)
                    chunk_indices.append(chunk_idx)
                    sub_idx += 1
                else:
                    break

    if not chunk_files:
        print("No chunk files found for categorization")
        return [], {}

    # Time indices for analysis
    pre_stim_mask = time_test < stimulus_start
    during_stim_mask = (time_test >= stimulus_start) & (time_test <= stimulus_end)

    # Find time index for 60ms after stimulus onset (for membrane potential check)
    check_time = stimulus_start + config.VOLTAGE_CHECK_TIME - stimulus_start  # 60.0 ms after stimulus
    check_time_idx = np.argmin(np.abs(time_test - check_time))
    print(f"Checking membrane potential at t={time_test[check_time_idx]:.2f} ms (target: {check_time:.2f} ms)")

    # Initialize category counts
    categories_count = {
        'no_ap': 0,
        'single_ap_after_stim': 0,
        'multiple_ap_during_stim': 0,
        'ap_before_stim': 0,
        'failed_voltage_criterion': 0  # Tracks traces that meet AP criteria but fail voltage check
    }

    # Track indices of single AP traces
    single_ap_indices = []
    total_traces = 0

    for file_idx, chunk_file in enumerate(chunk_files):
        try:
            chunk = np.load(chunk_file)
            num_traces = chunk.shape[0]

            for i in range(num_traces):
                total_traces += 1
                trace = chunk[i, :]

                # Skip if trace contains NaN values
                if np.any(np.isnan(trace)):
                    continue

                # Detect action potentials by finding upward threshold crossings
                above_threshold = trace > ap_threshold
                threshold_crossings = np.diff(above_threshold.astype(int)) > 0
                ap_indices = np.where(threshold_crossings)[0]

                # Count APs in different time periods
                ap_before_stim = np.sum(ap_indices < np.sum(pre_stim_mask))
                ap_during_stim = np.sum((ap_indices >= np.sum(pre_stim_mask)) &
                                       (ap_indices < np.sum(pre_stim_mask) + np.sum(during_stim_mask)))
                total_aps = len(ap_indices)

                # Categorize the trace
                if ap_before_stim > 0:
                    category = 'ap_before_stim'
                elif total_aps == 0:
                    category = 'no_ap'
                elif ap_during_stim == 1 and total_aps == 1:
                    # Check additional voltage criterion: V at check_time < -40 mV
                    V_at_check_time = trace[check_time_idx]
                    if V_at_check_time < config.VOLTAGE_CHECK_THRESHOLD:
                        category = 'single_ap_after_stim'
                        # Store the chunk index and trace index for this single AP trace
                        single_ap_indices.append((file_idx, i))
                    else:
                        category = 'failed_voltage_criterion'
                elif ap_during_stim >= 2:
                    category = 'multiple_ap_during_stim'
                else:
                    # Edge case: AP after stimulus period
                    if total_aps == 1:
                        # Check additional voltage criterion
                        V_at_check_time = trace[check_time_idx]
                        if V_at_check_time < config.VOLTAGE_CHECK_THRESHOLD:
                            category = 'single_ap_after_stim'
                            single_ap_indices.append((file_idx, i))
                        else:
                            category = 'failed_voltage_criterion'
                    else:
                        category = 'multiple_ap_during_stim'

                categories_count[category] += 1

                # Progress update
                if total_traces % 10000 == 0:
                    print(f"  Processed {total_traces} traces, found {len(single_ap_indices)} single AP traces...")

        except Exception as e:
            print(f"Warning: Could not read {chunk_file} for categorization: {e}")

    print(f"\nCategorization complete. Processed {total_traces} traces.")
    print(f"Found {len(single_ap_indices)} traces with single AP after stimulus onset")

    # Print summary
    print("\nTrace categorization results:")
    for category, count in categories_count.items():
        percentage = (count / total_traces) * 100 if total_traces > 0 else 0
        print(f"  {category}: {count} traces ({percentage:.1f}%)")

    return single_ap_indices, categories_count


def create_masked_data(save_path: str, num_chunks: int, single_ap_indices: List[Tuple[int, int]],
                      nT: int, filtered_save_path: str = config.FILTERED_SAVE_PATH_DEFAULT) -> Tuple[Optional[str], int]:
    """
    Create masked dataset where non-single-AP traces are replaced with mean of single-AP traces.
    This preserves the Saltelli sample structure required for Sobol analysis.

    Parameters
    ----------
    save_path : str
        Directory containing original chunk files
    num_chunks : int
        Number of original chunk files
    single_ap_indices : List[Tuple[int, int]]
        List of tuples (chunk_file_idx, trace_idx_in_chunk) for single AP traces
    nT : int
        Number of time points
    filtered_save_path : str
        Directory to save masked chunk files

    Returns
    -------
    filtered_save_path : str or None
        Path to masked data directory (None if failed)
    num_filtered_chunks : int
        Number of chunk files created (same as input)
    """
    print(f"\nCreating masked dataset (preserving Saltelli structure)...")
    print(f"Single AP traces: {len(single_ap_indices)}")

    if not os.path.exists(filtered_save_path):
        os.makedirs(filtered_save_path)

    # Find all chunk files and track their original names
    chunk_files = []
    chunk_names = []  # Track original chunk identifiers
    for chunk_idx in range(num_chunks):
        main_chunk_file = os.path.join(save_path, f'chunk_{chunk_idx}.npy')
        if os.path.exists(main_chunk_file):
            chunk_files.append(main_chunk_file)
            chunk_names.append(f'chunk_{chunk_idx}.npy')
        else:
            # Look for sub-chunks
            sub_idx = 0
            while True:
                sub_chunk_file = os.path.join(save_path, f'chunk_{chunk_idx}_sub_{sub_idx}.npy')
                if os.path.exists(sub_chunk_file):
                    chunk_files.append(sub_chunk_file)
                    chunk_names.append(f'chunk_{chunk_idx}_sub_{sub_idx}.npy')
                    sub_idx += 1
                else:
                    break

    # Create a mask array indicating which traces are single AP
    # First, we need to create a global index mapping
    indices_by_chunk = {}
    for file_idx, trace_idx in single_ap_indices:
        if file_idx not in indices_by_chunk:
            indices_by_chunk[file_idx] = set()
        indices_by_chunk[file_idx].add(trace_idx)

    # First pass: compute mean of single AP traces
    print("Computing mean of single AP traces...")
    single_ap_sum = np.zeros(nT, dtype=np.float64)
    single_ap_count = 0

    for file_idx in sorted(indices_by_chunk.keys()):
        try:
            chunk = np.load(chunk_files[file_idx])
            trace_indices = list(indices_by_chunk[file_idx])

            # Sum single AP traces
            for idx in trace_indices:
                if idx < chunk.shape[0] and not np.any(np.isnan(chunk[idx, :])):
                    single_ap_sum += chunk[idx, :]
                    single_ap_count += 1

        except Exception as e:
            print(f"Warning: Could not read chunk file {file_idx}: {e}")

    if single_ap_count == 0:
        print("ERROR: No valid single AP traces found!")
        return None, 0

    mean_single_ap = single_ap_sum / single_ap_count
    print(f"Computed mean from {single_ap_count} single AP traces")

    # Second pass: create masked chunks
    print("Creating masked chunks...")
    for file_idx, (chunk_file, chunk_name) in enumerate(zip(chunk_files, chunk_names)):
        try:
            chunk = np.load(chunk_file)
            masked_chunk = chunk.copy()

            # Get set of single AP indices for this chunk
            single_ap_set = indices_by_chunk.get(file_idx, set())

            # Replace non-single-AP traces with mean
            num_replaced = 0
            for i in range(chunk.shape[0]):
                if i not in single_ap_set:
                    masked_chunk[i, :] = mean_single_ap
                    num_replaced += 1

            # Save masked chunk with same name as original (preserves structure)
            masked_file = os.path.join(filtered_save_path, chunk_name)
            np.save(masked_file, masked_chunk.astype(np.float32))
            print(f"  {chunk_name}: kept {len(single_ap_set)} single AP, replaced {num_replaced} with mean")

        except Exception as e:
            print(f"Warning: Could not process {chunk_name}: {e}")

    print(f"\nCreated masked dataset in {filtered_save_path}")
    print(f"Structure preserved: {len(chunk_files)} chunks, same sample count per chunk")

    # Validate total sample count
    total_samples_original = 0
    total_samples_masked = 0
    for chunk_file in chunk_files:
        chunk = np.load(chunk_file)
        total_samples_original += chunk.shape[0]
    for chunk_name in chunk_names:
        masked_file = os.path.join(filtered_save_path, chunk_name)
        if os.path.exists(masked_file):
            chunk = np.load(masked_file)
            total_samples_masked += chunk.shape[0]

    print(f"Validation: Original samples = {total_samples_original}, Masked samples = {total_samples_masked}")
    if total_samples_original != total_samples_masked:
        print(f"WARNING: Sample count mismatch! This will cause Sobol analysis to fail.")
        return None, 0

    # Additional validation: Check if sample count matches Saltelli structure
    # For Saltelli with calc_second_order=True: samples = N * (2*D + 2)
    # We should have exactly this many samples
    print(f"\nValidating Saltelli structure:")
    print(f"  Total samples in masked data: {total_samples_masked}")
    print(f"  This validation will be checked again during Sobol analysis")

    # Return num_chunks (number of main chunk indices), not len(chunk_files)
    # This ensures consistency with the skip_filtering logic
    return filtered_save_path, num_chunks


def worker_sobol_batch(args: Tuple[int, List[int], str, int, Dict[str, Any]]) -> List[Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    Worker function to perform Sobol analysis for a batch of time points.
    Uses TRANSPOSED data (time × samples) with memory-mapping for efficient, low-memory access.
    Each process handles an entire batch sequentially to avoid SALib thread-safety issues.

    Parameters
    ----------
    args : tuple
        (batch_idx, time_indices, save_path, num_chunks, problem)

    Returns
    -------
    List[Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]]
        List of (time_idx, S1_values, ST_values) for each time point
    """
    from SALib.analyze import sobol

    batch_idx, time_indices, save_path, num_chunks, problem = args

    print(f"  Process handling batch {batch_idx + 1} with {len(time_indices)} time points...")

    # Find all TRANSPOSED chunk files
    transposed_files = []
    for chunk_idx in range(num_chunks):
        main_transposed = os.path.join(save_path, f'chunk_{chunk_idx}_transposed.npy')
        if os.path.exists(main_transposed):
            transposed_files.append(main_transposed)
        else:
            # Look for transposed sub-chunks
            sub_idx = 0
            while True:
                sub_transposed = os.path.join(save_path, f'chunk_{chunk_idx}_sub_{sub_idx}_transposed.npy')
                if os.path.exists(sub_transposed):
                    transposed_files.append(sub_transposed)
                    sub_idx += 1
                else:
                    break

    if not transposed_files:
        raise FileNotFoundError(f"No transposed chunk files found in {save_path}. Run transpose_chunks_for_sobol() first!")

    # Open all transposed chunks with memory-mapping (minimal RAM usage)
    # Memory-mapped files are cached by the OS and shared across processes
    print(f"Memory-mapping {len(transposed_files)} transposed chunks...")
    transposed_chunks = []
    total_samples = 0

    for trans_file in transposed_files:
        try:
            chunk = np.load(trans_file, mmap_mode='r')  # Memory-mapped read (time × samples)
            transposed_chunks.append(chunk)
            total_samples += chunk.shape[1]  # Samples are now in columns
        except Exception as e:
            print(f"  Warning: Could not map {trans_file}: {e}")

    print(f"Mapped {total_samples} samples. Processing {len(time_indices)} time points...")

    # Validate sample count against problem definition
    # For Saltelli with calc_second_order=True: samples = N * (2*D + 2)
    expected_samples = problem['num_vars'] * 2 + 2  # This is (2*D + 2)
    # We don't know N directly, but total_samples should be divisible by expected_samples
    if total_samples % expected_samples != 0:
        print(f"  ERROR: Sample count ({total_samples}) is not divisible by (2*D+2) = {expected_samples}")
        print(f"  This indicates a structural problem with the data!")
        print(f"  Expected: N * {expected_samples} for some integer N")
    else:
        N_actual = total_samples // expected_samples
        print(f"  Detected N = {N_actual} (total samples = {total_samples} = {N_actual} × {expected_samples})")

    # Initialize results for this batch
    batch_results = []

    for ti in time_indices:
        try:
            # Collect data for time point ti from all transposed chunks
            # Memory-mapped access: OS loads only needed rows from disk into cache
            all_V_ti = np.zeros(total_samples, dtype=np.float32)
            sample_idx = 0

            for chunk in transposed_chunks:
                chunk_samples = chunk.shape[1]
                # This row access is FAST with transposed data - single sequential read
                all_V_ti[sample_idx:sample_idx + chunk_samples] = chunk[ti, :].astype(np.float32)
                sample_idx += chunk_samples

            # Replace NaN values with data from previous valid sample
            # This ensures SALib receives exactly the expected number of samples
            nan_mask = np.isnan(all_V_ti)
            if np.any(nan_mask):
                nan_indices = np.where(nan_mask)[0]
                for nan_idx in nan_indices:
                    # Find the nearest previous valid sample
                    replacement_idx = nan_idx - 1
                    while replacement_idx >= 0 and np.isnan(all_V_ti[replacement_idx]):
                        replacement_idx -= 1

                    if replacement_idx >= 0:
                        # Replace with previous valid sample
                        all_V_ti[nan_idx] = all_V_ti[replacement_idx]
                    else:
                        # If no previous valid sample, search forward
                        replacement_idx = nan_idx + 1
                        while replacement_idx < len(all_V_ti) and np.isnan(all_V_ti[replacement_idx]):
                            replacement_idx += 1

                        if replacement_idx < len(all_V_ti):
                            all_V_ti[nan_idx] = all_V_ti[replacement_idx]
                        else:
                            # Last resort: use mean of all valid values
                            valid_values = all_V_ti[~np.isnan(all_V_ti)]
                            if len(valid_values) > 0:
                                all_V_ti[nan_idx] = np.mean(valid_values)
                            else:
                                # If everything is NaN, use a default value
                                all_V_ti[nan_idx] = -65.0  # Resting potential

            Y_t = all_V_ti

            # Check if we have enough valid data for SALib
            if len(Y_t) >= 2 and not np.any(np.isnan(Y_t)):
                # Variance-based (Sobol') analysis
                # Disable internal parallelization to avoid conflicts with outer parallelization
                # Reduce bootstrap resamples significantly for very large N to speed up analysis
                num_resamples = config.BOOTSTRAP_RESAMPLES_DEFAULT if total_samples > 700000 else config.BOOTSTRAP_RESAMPLES_DEFAULT if total_samples > 500000 else config.BOOTSTRAP_RESAMPLES_DEFAULT
                Si = sobol.analyze(
                    problem,
                    Y=Y_t,
                    calc_second_order=True,
                    conf_level=config.CONF_LEVEL,
                    print_to_console=False,
                    parallel=False,  # CRITICAL: Disable to prevent nested parallelization (we already parallelize at batch level)
                    num_resamples=num_resamples
                )
                batch_results.append((ti, Si['S1'], Si['ST']))
            else:
                batch_results.append((ti, None, None))

            # Explicitly free memory for this time point
            del all_V_ti, Y_t

        except Exception as e:
            print(f"  Sobol analysis failed for time point {ti} with error: {e}")
            batch_results.append((ti, None, None))

    print(f"  Batch {batch_idx + 1} completed with {len([r for r in batch_results if r[1] is not None])} successful analyses")
    return batch_results


def batch_sobol_analysis_from_disk(save_path: str, num_chunks: int, problem: Dict[str, Any],
                                  num_processes: int, batch_size: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Sobol analysis in batches, distributing batches across processors.
    Each processor handles an entire batch sequentially to avoid SALib thread-safety issues.

    Parameters
    ----------
    save_path : str
        Directory containing transposed chunk files
    num_chunks : int
        Number of chunk files
    problem : Dict[str, Any]
        SALib problem definition
    num_processes : int
        Number of parallel processes to use
    batch_size : int
        Number of time points to process in each batch

    Returns
    -------
    S1_time, ST_time : np.ndarray
        First-order and total-order Sobol indices
    """
    # Find all transposed chunk files
    transposed_files = []
    for chunk_idx in range(num_chunks):
        main_transposed = os.path.join(save_path, f'chunk_{chunk_idx}_transposed.npy')
        if os.path.exists(main_transposed):
            transposed_files.append(main_transposed)
        else:
            # Look for transposed sub-chunks
            sub_idx = 0
            while True:
                sub_transposed = os.path.join(save_path, f'chunk_{chunk_idx}_sub_{sub_idx}_transposed.npy')
                if os.path.exists(sub_transposed):
                    transposed_files.append(sub_transposed)
                    sub_idx += 1
                else:
                    break

    if not transposed_files:
        raise FileNotFoundError(f"No transposed chunk files found in {save_path}")

    print(f"Found {len(transposed_files)} transposed chunk files")

    # Determine nT and total_samples from first available transposed chunk file
    first_chunk = np.load(transposed_files[0], mmap_mode='r')
    nT = first_chunk.shape[0]
    total_samples = 0

    for trans_file in transposed_files:
        try:
            chunk = np.load(trans_file, mmap_mode='r')  # Memory-mapped read (time × samples)
            total_samples += chunk.shape[1]
        except Exception as e:
            print(f"Warning: Could not read {trans_file}: {e}")

    print(f"Total samples: {total_samples}, Time points: {nT}")
    print(f"Using batch-wise parallelization with {num_processes} processes (SALib-friendly approach)")

    S1_time = np.zeros((nT, problem['num_vars']))
    ST_time = np.zeros((nT, problem['num_vars']))

    # Create batches of time indices
    time_indices = list(range(nT))
    num_batches = int(np.ceil(nT / batch_size))

    # Split time indices into batches for parallel processing
    batch_args = []
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, nT)
        batch_time_indices = time_indices[batch_start:batch_end]

        # Each process gets: (batch_idx, time_indices, save_path, num_chunks, problem)
        batch_args.append((batch_idx, batch_time_indices, save_path, num_chunks, problem.copy()))

    print(f"Processing Sobol analysis in {num_batches} batches distributed across {num_processes} processes...")
    print(f"Using concurrent.futures.ProcessPoolExecutor for real-time progress tracking...")

    # Process batches in parallel with ProcessPoolExecutor for better progress tracking
    successful_analyses = 0
    completed_batches = 0

    executor = None
    try:
        executor = ProcessPoolExecutor(max_workers=num_processes)

        # Submit all batch jobs and create a mapping of futures to batch info
        future_to_batch = {
            executor.submit(worker_sobol_batch, args): (batch_idx, len(args[1]))
            for batch_idx, args in enumerate(batch_args)
        }

        # Process results as they complete (provides immediate feedback)
        for future in as_completed(future_to_batch):
            batch_idx, num_time_points = future_to_batch[future]
            completed_batches += 1

            try:
                batch_results = future.result()

                # Combine results from this batch
                batch_successes = 0
                for ti, S1, ST in batch_results:
                    if S1 is not None and ST is not None:
                        S1_time[ti, :] = S1
                        ST_time[ti, :] = ST
                        successful_analyses += 1
                        batch_successes += 1

                # Print immediate progress feedback
                print(f"✓ Batch {batch_idx + 1}/{num_batches} completed: {batch_successes}/{num_time_points} successful analyses | Total progress: {completed_batches}/{num_batches} batches ({100*completed_batches/num_batches:.1f}%)")

            except Exception as e:
                print(f"✗ Batch {batch_idx + 1}/{num_batches} failed with error: {e}")

    except Exception as e:
        print(f"Error during parallel Sobol analysis: {e}")
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    print(f"Sobol analysis completed: {successful_analyses}/{nT} time points successful")

    return S1_time, ST_time


def compute_statistics_from_disk(save_path: str, num_chunks: int, time_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute statistics (mean, std) from chunk files or transposed files without loading all data.
    Handles NaN values by replacing them with previous valid samples.

    Parameters
    ----------
    save_path : str
        Directory containing chunk files or transposed files
    num_chunks : int
        Number of chunk files
    time_test : np.ndarray
        Time vector

    Returns
    -------
    mean_V, std_V : np.ndarray
        Mean and standard deviation of voltage traces
    """
    # Check if we have transposed files (preferred) or original chunk files
    transposed_files = sorted(glob.glob(os.path.join(save_path, 'chunk_*_transposed.npy')))

    if transposed_files:
        # Use transposed files (time × samples format) - more efficient
        print(f"Computing statistics from {len(transposed_files)} transposed files...")

        # Get dimensions
        first_chunk = np.load(transposed_files[0], mmap_mode='r')
        nT = first_chunk.shape[0]
        total_samples = sum(np.load(f, mmap_mode='r').shape[1] for f in transposed_files)

        print(f"Time points: {nT}, Total samples: {total_samples}")

        # Compute statistics time point by time point with NaN handling
        mean_V = np.zeros(nT)
        var_V = np.zeros(nT)

        # Memory-map all files once at the beginning
        print("Memory-mapping transposed files...")
        mapped_chunks = [np.load(f, mmap_mode='r') for f in transposed_files]
        print(f"Mapped {len(mapped_chunks)} files successfully")

        for ti in range(nT):
            if ti % 500 == 0:  # More frequent updates
                print(f"  Progress: {ti}/{nT} ({100*ti/nT:.1f}%)")

            # Collect data for this time point from all transposed chunks
            all_V_ti = []
            for chunk in mapped_chunks:
                all_V_ti.append(chunk[ti, :])

            all_V_ti = np.concatenate(all_V_ti).astype(np.float32)

            # Replace NaN values with previous valid sample (same logic as Sobol worker)
            nan_mask = np.isnan(all_V_ti)
            if np.any(nan_mask):
                nan_indices = np.where(nan_mask)[0]
                for nan_idx in nan_indices:
                    replacement_idx = nan_idx - 1
                    while replacement_idx >= 0 and np.isnan(all_V_ti[replacement_idx]):
                        replacement_idx -= 1
                    if replacement_idx >= 0:
                        all_V_ti[nan_idx] = all_V_ti[replacement_idx]
                    else:
                        # If no previous valid sample, search forward
                        replacement_idx = nan_idx + 1
                        while replacement_idx < len(all_V_ti) and np.isnan(all_V_ti[replacement_idx]):
                            replacement_idx += 1
                        if replacement_idx < len(all_V_ti):
                            all_V_ti[nan_idx] = all_V_ti[replacement_idx]
                        else:
                            # Last resort: use -65.0 (resting potential)
                            all_V_ti[nan_idx] = -65.0

            # Compute statistics for this time point
            mean_V[ti] = np.mean(all_V_ti)
            var_V[ti] = np.var(all_V_ti)

        std_V = np.sqrt(var_V)
        print(f"Statistics computed successfully (no NaN values: {np.isnan(mean_V).sum() == 0})")

    else:
        # Fall back to original chunk files (samples × time format)
        chunk_files = []
        for chunk_idx in range(num_chunks):
            main_chunk_file = os.path.join(save_path, f'chunk_{chunk_idx}.npy')
            if os.path.exists(main_chunk_file):
                chunk_files.append(main_chunk_file)
            else:
                # Look for sub-chunks
                sub_idx = 0
                while True:
                    sub_chunk_file = os.path.join(save_path, f'chunk_{chunk_idx}_sub_{sub_idx}.npy')
                    if os.path.exists(sub_chunk_file):
                        chunk_files.append(sub_chunk_file)
                        sub_idx += 1
                    else:
                        break

        if not chunk_files:
            raise FileNotFoundError("No chunk files or transposed files found")

        # Get dimensions from first chunk
        first_chunk = np.load(chunk_files[0])
        nT = first_chunk.shape[1]

        # Count total samples
        total_samples = 0
        for chunk_file in chunk_files:
            try:
                chunk = np.load(chunk_file, mmap_mode='r')
                total_samples += chunk.shape[0]
            except Exception as e:
                print(f"Warning: Could not read {chunk_file} for variance calculation: {e}")

        print(f"Computing statistics from {total_samples} samples across {len(chunk_files)} chunk files...")

        # Compute mean with NaN handling (use nanmean instead of mean)
        mean_V = np.zeros(nT)
        print("Computing mean...")
        for idx, chunk_file in enumerate(chunk_files):
            if idx % 5 == 0:
                print(f"  Mean progress: {idx+1}/{len(chunk_files)} files")
            try:
                chunk = np.load(chunk_file)
                mean_V += np.nansum(chunk, axis=0)
            except Exception as e:
                print(f"Warning: Could not read {chunk_file} for mean calculation: {e}")
        mean_V /= total_samples
        print("Mean computed successfully")

        # Compute standard deviation with NaN handling
        var_V = np.zeros(nT)
        print("Computing variance...")
        for idx, chunk_file in enumerate(chunk_files):
            if idx % 5 == 0:
                print(f"  Variance progress: {idx+1}/{len(chunk_files)} files")
            try:
                chunk = np.load(chunk_file)
                var_V += np.nansum((chunk - mean_V)**2, axis=0)
            except Exception as e:
                print(f"Warning: Could not read {chunk_file} for variance calculation: {e}")
        std_V = np.sqrt(var_V / total_samples)
        print("Variance computed successfully")

    return mean_V, std_V


def save_sample_traces(save_path: str, num_chunks: int, output_file: str = config.ALL_V_SAMPLE_CSV,
                      num_samples: int = 1000) -> None:
    """
    Save a sample of voltage traces from transposed chunk files.

    Parameters
    ----------
    save_path : str
        Directory containing transposed chunk files
    num_chunks : int
        Number of chunk files
    output_file : str
        Output CSV file name
    num_samples : int
        Number of samples to save
    """
    print(f"Saving {num_samples} sample traces...")

    # Find all transposed chunk files
    transposed_files = []
    for chunk_idx in range(num_chunks):
        main_transposed = os.path.join(save_path, f'chunk_{chunk_idx}_transposed.npy')
        if os.path.exists(main_transposed):
            transposed_files.append(main_transposed)
        else:
            # Look for transposed sub-chunks
            sub_idx = 0
            while True:
                sub_transposed = os.path.join(save_path, f'chunk_{chunk_idx}_sub_{sub_idx}_transposed.npy')
                if os.path.exists(sub_transposed):
                    transposed_files.append(sub_transposed)
                    sub_idx += 1
                else:
                    break

    if not transposed_files:
        print("No transposed chunk files found for sampling")
        return

    # Load first transposed chunk to get dimensions
    first_chunk = np.load(transposed_files[0], mmap_mode='r')
    nT = first_chunk.shape[0]

    # Collect samples from transposed files (time × samples format)
    samples_collected = 0
    sample_data = []

    for trans_file in transposed_files:
        if samples_collected >= num_samples:
            break

        try:
            chunk = np.load(trans_file, mmap_mode='r')  # (time, samples)
            samples_in_chunk = chunk.shape[1]
            samples_to_take = min(num_samples - samples_collected, samples_in_chunk)
            
            # Take samples from this chunk (all time points, selected samples)
            selected_samples = chunk[:, :samples_to_take]  # (time, samples_to_take)
            # Transpose to (samples, time) format for saving
            sample_data.append(selected_samples.T)
            samples_collected += samples_to_take
        except Exception as e:
            print(f"Warning: Could not read {trans_file} for sampling: {e}")

    if sample_data:
        # Combine and save
        all_samples = np.vstack(sample_data)
        np.savetxt(output_file, all_samples, delimiter=',')
        print(f"Saved {samples_collected} samples to {output_file}")
    else:
        print("No samples could be collected")