"""
Core analysis module for the Hodgkin-Huxley cable model.

Contains analysis functions that process simulation data and return results
without creating plots.
"""

import numpy as np
import os


def analyze_resting_potentials(save_path, num_chunks):
    """
    Analyze the first data point from each simulation (resting membrane potential).

    Parameters:
    -----------
    save_path : str
        Directory containing chunk files
    num_chunks : int
        Number of chunk files

    Returns:
    --------
    resting_potentials : numpy array
        Array of resting potential values
    """
    print("Analyzing resting membrane potentials...")

    # Find all injection chunk files
    chunk_files = []
    for chunk_idx in range(num_chunks):
        main_chunk_file = os.path.join(save_path, f'chunk_{chunk_idx}_injection.npy')
        if os.path.exists(main_chunk_file):
            chunk_files.append(main_chunk_file)
        else:
            # Look for sub-chunks
            sub_idx = 0
            while True:
                sub_chunk_file = os.path.join(save_path, f'chunk_{chunk_idx}_sub_{sub_idx}_injection.npy')
                if os.path.exists(sub_chunk_file):
                    chunk_files.append(sub_chunk_file)
                    sub_idx += 1
                else:
                    break

    if not chunk_files:
        print("No chunk files found for analysis")
        return np.array([])

    # Collect first data point (resting potential) from each simulation
    resting_potentials = []

    for chunk_file in chunk_files:
        try:
            chunk = np.load(chunk_file)
            # Extract first time point (column 0) for all simulations in this chunk
            resting_vals = chunk[:, 0]  # Shape: (num_simulations,)
            # Filter out NaN values
            valid_vals = resting_vals[~np.isnan(resting_vals)]
            resting_potentials.extend(valid_vals)
        except Exception as e:
            print(f"Warning: Could not read {chunk_file} for resting potential analysis: {e}")

    resting_potentials = np.array(resting_potentials)
    print(f"Collected {len(resting_potentials)} resting potential values")

    return resting_potentials


def categorize_traces(save_path, num_chunks, time_test, param_values=None, stimulus_start=10.0, stimulus_end=80.0,
                     ap_threshold=-20.0):
    """
    Categorize voltage traces based on action potential patterns:
    1. No action potential during entire simulation
    2. Single action potential after stimulus onset, no more APs
    3. Multiple action potentials during stimulus period
    4. Action potential before stimulus onset

    Parameters:
    -----------
    save_path : str
        Directory containing chunk files
    num_chunks : int
        Number of chunk files
    time_test : array
        Time vector
    param_values : array, optional
        Parameter sets used for simulations (for parameter analysis)
    stimulus_start : float
        Start time of stimulus (ms)
    stimulus_end : float
        End time of stimulus (ms)
    ap_threshold : float
        Voltage threshold for detecting action potentials (mV)

    Returns:
    --------
    categories : dict
        Dictionary with counts, representative traces, and parameter indices for each category
    """
    print("Categorizing voltage traces...")

    # Find all injection chunk files
    chunk_files = []
    for chunk_idx in range(num_chunks):
        main_chunk_file = os.path.join(save_path, f'chunk_{chunk_idx}_injection.npy')
        if os.path.exists(main_chunk_file):
            chunk_files.append(main_chunk_file)
        else:
            # Look for sub-chunks
            sub_idx = 0
            while True:
                sub_chunk_file = os.path.join(save_path, f'chunk_{chunk_idx}_sub_{sub_idx}_injection.npy')
                if os.path.exists(sub_chunk_file):
                    chunk_files.append(sub_chunk_file)
                    sub_idx += 1
                else:
                    break

    if not chunk_files:
        print("No chunk files found for categorization")
        return {}

    # Time indices for analysis
    pre_stim_mask = time_test < stimulus_start
    during_stim_mask = (time_test >= stimulus_start) & (time_test <= stimulus_end)
    post_stim_mask = time_test > stimulus_start

    # Initialize categories
    categories = {
        'no_ap': {'count': 0, 'traces': [], 'param_indices': []},
        'single_ap_propagated': {'count': 0, 'traces': [], 'param_indices': []},
        'single_ap_failed': {'count': 0, 'traces': [], 'param_indices': []},
        'multiple_ap_during_stim': {'count': 0, 'traces': [], 'param_indices': []},
        'ap_before_stim': {'count': 0, 'traces': [], 'param_indices': []}
    }

    total_traces = 0
    max_examples = 5  # Store up to 5 representative traces per category

    # Also find corresponding distal chunk files for propagation analysis
    chunk_files_distal = []
    for chunk_file_inj in chunk_files:
        chunk_file_dist = chunk_file_inj.replace('_injection.npy', '_distal.npy')
        chunk_files_distal.append(chunk_file_dist)

    for chunk_file, chunk_file_dist in zip(chunk_files, chunk_files_distal):
        try:
            chunk = np.load(chunk_file)
            chunk_dist = np.load(chunk_file_dist)
            num_traces = chunk.shape[0]

            for i in range(num_traces):
                global_idx = total_traces  # Track global index for parameter lookup
                total_traces += 1
                trace = chunk[i, :]
                trace_dist = chunk_dist[i, :]

                # Skip if trace contains NaN values
                if np.any(np.isnan(trace)) or np.any(np.isnan(trace_dist)):
                    continue

                # Detect action potentials by finding upward threshold crossings
                # This counts distinct APs, not all points above threshold
                above_threshold = trace > ap_threshold

                # Find where trace crosses threshold from below (upward crossings = AP initiation)
                # Use diff to find transitions: 0->1 (False->True)
                threshold_crossings = np.diff(above_threshold.astype(int)) > 0

                # Get indices of APs in different time periods
                ap_indices = np.where(threshold_crossings)[0]

                # Count APs in different time periods based on their initiation time
                ap_before_stim = np.sum(ap_indices < np.sum(pre_stim_mask))
                ap_during_stim = np.sum((ap_indices >= np.sum(pre_stim_mask)) &
                                       (ap_indices < np.sum(pre_stim_mask) + np.sum(during_stim_mask)))
                total_aps = len(ap_indices)

                # Categorize the trace based on the four categories:
                # Priority order: ap_before_stim > no_ap > single_ap > multiple_ap

                if ap_before_stim > 0:
                    # Category 4: AP generated before stimulus onset (0-10 ms)
                    category = 'ap_before_stim'

                elif total_aps == 0:
                    # Category 1: No action potential during entire simulation
                    category = 'no_ap'

                elif ap_during_stim == 1 and total_aps == 1:
                    # Category 2: Exactly one AP after stimulus onset
                    # Check if it propagated to distal site (X=8 cm)
                    # Measure distal AP amplitude in the region after stimulus start
                    V_rest_dist = trace_dist[0]
                    # Look for peak after stimulus start (where we expect the AP)
                    stim_start_idx = np.sum(pre_stim_mask)
                    V_peak_dist = np.max(trace_dist[stim_start_idx:])
                    amp_dist = V_peak_dist - V_rest_dist

                    if amp_dist >= 70.0:  # Successful propagation threshold
                        category = 'single_ap_propagated'
                    else:
                        category = 'single_ap_failed'

                elif ap_during_stim >= 2:
                    # Category 3: Sustained repetitive firing throughout stimulus period
                    # MUST have sustained repetitive firing, not just onset burst
                    # Check if APs are distributed throughout stimulus period
                    stim_start_idx = np.sum(pre_stim_mask)
                    stim_end_idx = np.sum(pre_stim_mask) + np.sum(during_stim_mask)
                    ap_during_indices = ap_indices[(ap_indices >= stim_start_idx) &
                                                   (ap_indices < stim_end_idx)]

                    # Get AP times during stimulus
                    ap_times = ap_during_indices * (time_test[1] - time_test[0])  # Convert to time
                    first_ap_time = ap_times[0]
                    last_ap_time = ap_times[-1]

                    # Require sustained firing: last AP must be at least 30ms after first AP
                    # This excludes onset bursts/doublets (which typically finish within 10-20ms)
                    firing_duration = last_ap_time - first_ap_time

                    if firing_duration >= 30.0:  # Sustained repetitive firing
                        category = 'multiple_ap_during_stim'
                    else:
                        # Onset burst only - treat as single AP with propagation check
                        V_rest_dist = trace_dist[0]
                        stim_start_idx = np.sum(pre_stim_mask)
                        V_peak_dist = np.max(trace_dist[stim_start_idx:])
                        amp_dist = V_peak_dist - V_rest_dist

                        if amp_dist >= 70.0:
                            category = 'single_ap_propagated'
                        else:
                            category = 'single_ap_failed'

                else:
                    # Edge case: AP exists but not during stimulus (e.g., after 80ms)
                    # We'll classify based on propagation
                    if total_aps == 1:
                        V_rest_dist = trace_dist[0]
                        stim_start_idx = np.sum(pre_stim_mask)
                        V_peak_dist = np.max(trace_dist[stim_start_idx:])
                        amp_dist = V_peak_dist - V_rest_dist

                        if amp_dist >= 70.0:
                            category = 'single_ap_propagated'
                        else:
                            category = 'single_ap_failed'
                    else:
                        # Multiple APs but not during stimulus - treat as single AP
                        V_rest_dist = trace_dist[0]
                        stim_start_idx = np.sum(pre_stim_mask)
                        V_peak_dist = np.max(trace_dist[stim_start_idx:])
                        amp_dist = V_peak_dist - V_rest_dist

                        if amp_dist >= 70.0:
                            category = 'single_ap_propagated'
                        else:
                            category = 'single_ap_failed'

                # Update count
                categories[category]['count'] += 1

                # Store parameter index for this trace
                categories[category]['param_indices'].append(global_idx)

                # Store representative trace (limit to max_examples per category)
                if len(categories[category]['traces']) < max_examples:
                    categories[category]['traces'].append(trace.copy())

                # Progress update
                if total_traces % 10000 == 0:
                    print(f"  Processed {total_traces} traces...")

        except Exception as e:
            print(f"Warning: Could not read {chunk_file} for categorization: {e}")

    print(f"Categorization complete. Processed {total_traces} traces.")

    # Print summary
    print("\nTrace categorization results:")
    for category, data in categories.items():
        percentage = (data['count'] / total_traces) * 100 if total_traces > 0 else 0
        print(f"  {category}: {data['count']} traces ({percentage:.1f}%)")

    return categories


def analyze_firing_frequency(save_path, num_chunks, time_test, stimulus_start=10.0,
                             stimulus_end=80.0, ap_threshold=-20.0):
    """
    Analyze firing frequency for traces with multiple action potentials during stimulus.
    Calculates the firing frequency (in Hz) based on mean inter-spike interval (ISI)
    for each simulation and creates a histogram.

    Parameters:
    -----------
    save_path : str
        Directory containing chunk files
    num_chunks : int
        Number of chunk files
    time_test : array
        Time vector
    stimulus_start : float
        Start time of stimulus (ms)
    stimulus_end : float
        End time of stimulus (ms)
    ap_threshold : float
        Voltage threshold for detecting action potentials (mV)

    Returns:
    --------
    firing_frequencies : numpy array
        Array of firing frequencies (Hz) calculated from mean ISI for all simulations
        with multiple APs during stimulus
    """
    print("Analyzing firing frequencies for multiple AP traces...")

    # Calculate time step from time vector
    dt = time_test[1] - time_test[0]

    # Find all injection chunk files
    chunk_files = []
    for chunk_idx in range(num_chunks):
        main_chunk_file = os.path.join(save_path, f'chunk_{chunk_idx}_injection.npy')
        if os.path.exists(main_chunk_file):
            chunk_files.append(main_chunk_file)
        else:
            # Look for sub-chunks
            sub_idx = 0
            while True:
                sub_chunk_file = os.path.join(save_path, f'chunk_{chunk_idx}_sub_{sub_idx}_injection.npy')
                if os.path.exists(sub_chunk_file):
                    chunk_files.append(sub_chunk_file)
                    sub_idx += 1
                else:
                    break

    if not chunk_files:
        print("No chunk files found for firing frequency analysis")
        return np.array([])

    # Time indices for analysis
    pre_stim_mask = time_test < stimulus_start
    during_stim_mask = (time_test >= stimulus_start) & (time_test <= stimulus_end)
    stimulus_duration_ms = stimulus_end - stimulus_start

    firing_frequencies = []
    total_traces = 0
    qualifying_traces = 0

    for chunk_file in chunk_files:
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

                # Get indices of APs
                ap_indices = np.where(threshold_crossings)[0]

                # Count APs in different time periods
                ap_before_stim = np.sum(ap_indices < np.sum(pre_stim_mask))
                ap_during_stim = np.sum((ap_indices >= np.sum(pre_stim_mask)) &
                                       (ap_indices < np.sum(pre_stim_mask) + np.sum(during_stim_mask)))

                # Only analyze traces with multiple APs during stimulus and no APs before
                # REQUIRE sustained repetitive firing, not just onset bursts
                if ap_before_stim == 0 and ap_during_stim >= 2:
                    # Get AP indices during stimulus
                    stim_start_idx = np.sum(pre_stim_mask)
                    stim_end_idx = np.sum(pre_stim_mask) + np.sum(during_stim_mask)
                    ap_during_indices = ap_indices[(ap_indices >= stim_start_idx) &
                                                   (ap_indices < stim_end_idx)]

                    # Calculate inter-spike intervals (ISI) in time units
                    # ap_indices are in array index units, convert to time (ms)
                    ap_times_ms = ap_during_indices * dt

                    # Check for sustained firing: last AP must be at least 30ms after first
                    firing_duration = ap_times_ms[-1] - ap_times_ms[0]

                    if firing_duration >= 30.0:  # Only include sustained repetitive firing
                        isis_ms = np.diff(ap_times_ms)

                        # Calculate mean ISI and convert to frequency (Hz)
                        # Frequency = 1 / mean_ISI (in seconds)
                        mean_isi_ms = np.mean(isis_ms)
                        frequency_hz = 1000.0 / mean_isi_ms

                        firing_frequencies.append(frequency_hz)
                        qualifying_traces += 1

                # Progress update
                if total_traces % 10000 == 0:
                    print(f"  Processed {total_traces} traces ({qualifying_traces} with sustained firing)...")

        except Exception as e:
            print(f"Warning: Could not read {chunk_file} for firing frequency analysis: {e}")

    firing_frequencies = np.array(firing_frequencies)
    print(f"Firing frequency analysis complete.")
    print(f"  Total traces processed: {total_traces}")
    print(f"  Traces with sustained repetitive firing: {qualifying_traces}")

    return firing_frequencies


def analyze_stable_potential_after_spike(save_path, num_chunks, time_test, measurement_time=80.0,
                                         stimulus_start=10.0, stimulus_end=80.0, ap_threshold=-20.0):
    """
    Analyze the stable membrane potential at a specific time point for traces with a single
    action potential after stimulus onset.

    Measures the membrane potential at measurement_time (default 80 ms, end of stimulus)
    for traces classified as having exactly one AP after stimulus onset with no subsequent APs.

    Parameters:
    -----------
    save_path : str
        Directory containing chunk files
    num_chunks : int
        Number of chunk files
    time_test : array
        Time vector
    measurement_time : float
        Time point to measure membrane potential (ms)
    stimulus_start : float
        Start time of stimulus (ms)
    stimulus_end : float
        End time of stimulus (ms)
    ap_threshold : float
        Voltage threshold for detecting action potentials (mV)

    Returns:
    --------
    stable_potentials : numpy array
        Array of membrane potentials (mV) at measurement_time for single AP traces
    """
    print(f"Analyzing stable membrane potential at {measurement_time} ms for single AP traces...")

    # Find the time index closest to measurement_time
    measurement_idx = np.argmin(np.abs(time_test - measurement_time))
    print(f"  Measuring at t={time_test[measurement_idx]:.3f} ms (index {measurement_idx})")

    # Find all injection chunk files
    chunk_files = []
    for chunk_idx in range(num_chunks):
        main_chunk_file = os.path.join(save_path, f'chunk_{chunk_idx}_injection.npy')
        if os.path.exists(main_chunk_file):
            chunk_files.append(main_chunk_file)
        else:
            # Look for sub-chunks
            sub_idx = 0
            while True:
                sub_chunk_file = os.path.join(save_path, f'chunk_{chunk_idx}_sub_{sub_idx}_injection.npy')
                if os.path.exists(sub_chunk_file):
                    chunk_files.append(sub_chunk_file)
                    sub_idx += 1
                else:
                    break

    if not chunk_files:
        print("No chunk files found for stable potential analysis")
        return np.array([])

    # Time masks for categorization
    pre_stim_mask = time_test < stimulus_start
    during_stim_mask = (time_test >= stimulus_start) & (time_test <= stimulus_end)

    stable_potentials = []
    total_traces = 0
    qualifying_traces = 0

    for chunk_file in chunk_files:
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

                # Get indices of APs
                ap_indices = np.where(threshold_crossings)[0]

                # Count APs in different time periods
                ap_before_stim = np.sum(ap_indices < np.sum(pre_stim_mask))
                ap_during_stim = np.sum((ap_indices >= np.sum(pre_stim_mask)) &
                                       (ap_indices < np.sum(pre_stim_mask) + np.sum(during_stim_mask)))
                total_aps = len(ap_indices)

                # Only analyze traces with single AP after stimulus onset and no APs before
                if ap_before_stim == 0 and ap_during_stim == 1 and total_aps == 1:
                    # Get membrane potential at measurement time
                    stable_potential = trace[measurement_idx]
                    stable_potentials.append(stable_potential)
                    qualifying_traces += 1

                # Progress update
                if total_traces % 10000 == 0:
                    print(f"  Processed {total_traces} traces ({qualifying_traces} with single AP)...")

        except Exception as e:
            print(f"Warning: Could not read {chunk_file} for stable potential analysis: {e}")

    stable_potentials = np.array(stable_potentials)
    print(f"Stable potential analysis complete.")
    print(f"  Total traces processed: {total_traces}")
    print(f"  Traces with single AP: {qualifying_traces}")

    return stable_potentials


def analyze_ap_propagation(save_path, num_chunks, time_test, stimulus_start=10.0, stimulus_end=80.0,
                           ap_threshold=-20.0, failure_threshold=10.0):
    """
    Analyze action potential propagation from injection site to distal site
    for traces with a single AP after stimulus onset.

    Measures:
    1. AP amplitude at injection site
    2. AP amplitude at distal site
    3. Propagation speed (m/s) for successfully propagated APs
    4. Success vs failure rate

    Parameters:
    -----------
    save_path : str
        Directory containing chunk files
    num_chunks : int
        Number of chunk files
    time_test : array
        Time vector
    stimulus_start : float
        Start time of stimulus (ms)
    stimulus_end : float
        End time of stimulus (ms)
    ap_threshold : float
        Voltage threshold for detecting action potentials (mV)
    failure_threshold : float
        Minimum AP amplitude at distal site to be considered successful propagation (mV)

    Returns:
    --------
    dict with keys: 'amplitudes_proximal', 'amplitudes_distal', 'speeds', 'summary'
    """
    print("\n" + "="*80)
    print("ANALYZING ACTION POTENTIAL PROPAGATION")
    print("="*80)

    # Find all chunk files (separate for injection and distal sites)
    chunk_files_injection = []
    chunk_files_distal = []

    for chunk_idx in range(num_chunks):
        main_inj = os.path.join(save_path, f'chunk_{chunk_idx}_injection.npy')
        main_dist = os.path.join(save_path, f'chunk_{chunk_idx}_distal.npy')

        if os.path.exists(main_inj) and os.path.exists(main_dist):
            chunk_files_injection.append(main_inj)
            chunk_files_distal.append(main_dist)
        else:
            # Look for sub-chunks
            sub_idx = 0
            while True:
                sub_inj = os.path.join(save_path, f'chunk_{chunk_idx}_sub_{sub_idx}_injection.npy')
                sub_dist = os.path.join(save_path, f'chunk_{chunk_idx}_sub_{sub_idx}_distal.npy')
                if os.path.exists(sub_inj) and os.path.exists(sub_dist):
                    chunk_files_injection.append(sub_inj)
                    chunk_files_distal.append(sub_dist)
                    sub_idx += 1
                else:
                    break

    if not chunk_files_injection:
        print("ERROR: No chunk files found for propagation analysis")
        print("Make sure simulations have been run and chunk files exist")
        return None

    print(f"Found {len(chunk_files_injection)} paired chunk files for analysis")

    # Time masks for categorization
    pre_stim_mask = time_test < stimulus_start
    during_stim_mask = (time_test >= stimulus_start) & (time_test <= stimulus_end)
    dt = time_test[1] - time_test[0]

    # Storage for results
    amplitudes_proximal = []  # AP amplitude at injection site
    amplitudes_distal = []    # AP amplitude at distal site
    propagation_speeds = []   # Speed in m/s for successful propagation

    # Track absolute peak voltages to verify V <= E_Na
    absolute_peaks_proximal = []
    absolute_peaks_distal = []

    # Storage for plotting all single AP traces
    all_single_ap_traces_inj = []   # All traces at injection site
    all_single_ap_traces_dist = []  # All traces at distal site

    total_traces = 0
    single_ap_traces = 0
    successful_propagation = 0
    failed_propagation = 0

    print("Processing chunk files...")

    for chunk_inj_file, chunk_dist_file in zip(chunk_files_injection, chunk_files_distal):
        try:
            chunk_inj = np.load(chunk_inj_file)
            chunk_dist = np.load(chunk_dist_file)
            num_traces = chunk_inj.shape[0]

            for i in range(num_traces):
                total_traces += 1
                trace_inj = chunk_inj[i, :]
                trace_dist = chunk_dist[i, :]

                # Skip if traces contain NaN values
                if np.any(np.isnan(trace_inj)) or np.any(np.isnan(trace_dist)):
                    continue

                # Detect action potentials at injection site
                above_threshold = trace_inj > ap_threshold
                threshold_crossings = np.diff(above_threshold.astype(int)) > 0
                ap_indices = np.where(threshold_crossings)[0]

                # Count APs in different time periods
                ap_before_stim = np.sum(ap_indices < np.sum(pre_stim_mask))
                ap_during_stim = np.sum((ap_indices >= np.sum(pre_stim_mask)) &
                                       (ap_indices < np.sum(pre_stim_mask) + np.sum(during_stim_mask)))
                total_aps = len(ap_indices)

                # Only analyze traces with single AP after stimulus onset (SELECTION CRITERION)
                if ap_before_stim == 0 and ap_during_stim == 1 and total_aps == 1:
                    single_ap_traces += 1

                    # Store traces for plotting
                    all_single_ap_traces_inj.append(trace_inj)
                    all_single_ap_traces_dist.append(trace_dist)

                    # Get resting potentials (first time point)
                    V_rest_inj = trace_inj[0]
                    V_rest_dist = trace_dist[0]

                    # Measure AP amplitude at injection site
                    V_peak_inj = np.max(trace_inj[np.sum(pre_stim_mask):])
                    amp_inj = V_peak_inj - V_rest_inj
                    amplitudes_proximal.append(amp_inj)
                    absolute_peaks_proximal.append(V_peak_inj)  # Track absolute peak

                    # Measure AP amplitude at distal site
                    V_peak_dist = np.max(trace_dist[np.sum(pre_stim_mask):])
                    amp_dist = V_peak_dist - V_rest_dist
                    amplitudes_distal.append(amp_dist)
                    absolute_peaks_distal.append(V_peak_dist)  # Track absolute peak

                    # Determine if AP propagated successfully
                    if amp_dist >= failure_threshold:
                        successful_propagation += 1

                        # Calculate propagation speed
                        # Find time of peak at both sites
                        idx_peak_inj = np.argmax(trace_inj[np.sum(pre_stim_mask):]) + np.sum(pre_stim_mask)
                        idx_peak_dist = np.argmax(trace_dist[np.sum(pre_stim_mask):]) + np.sum(pre_stim_mask)

                        t_peak_inj = time_test[idx_peak_inj]
                        t_peak_dist = time_test[idx_peak_dist]

                        # Propagation delay
                        delay_ms = t_peak_dist - t_peak_inj

                        # Only calculate speed if delay is positive and reasonable
                        if delay_ms > 0 and delay_ms < 50.0:  # Sanity check
                            from config import RECORDING_DISTANCE, INJECTION_DISTANCE
                            distance_cm = RECORDING_DISTANCE - INJECTION_DISTANCE
                            speed_cm_per_ms = distance_cm / delay_ms
                            speed_m_per_s = speed_cm_per_ms * 10.0  # Convert to m/s (cm/ms * 10 = m/s)
                            propagation_speeds.append(speed_m_per_s)
                    else:
                        failed_propagation += 1

                # Progress update
                if total_traces % 10000 == 0:
                    print(f"  Processed {total_traces} traces ({single_ap_traces} with single AP)...")

        except Exception as e:
            print(f"Warning: Could not read {chunk_inj_file} or {chunk_dist_file}: {e}")

    # Convert to numpy arrays
    amplitudes_proximal = np.array(amplitudes_proximal)
    amplitudes_distal = np.array(amplitudes_distal)
    propagation_speeds = np.array(propagation_speeds)
    all_single_ap_traces_inj = np.array(all_single_ap_traces_inj)
    all_single_ap_traces_dist = np.array(all_single_ap_traces_dist)

    print("\n" + "="*80)
    print("PROPAGATION ANALYSIS COMPLETE")
    print("="*80)
    print(f"Total traces processed: {total_traces:,}")
    print(f"Traces with single AP after stimulus: {single_ap_traces:,} ({100*single_ap_traces/total_traces:.1f}%)")
    print(f"\nPropagation Results:")
    print(f"  Successful propagation: {successful_propagation:,} ({100*successful_propagation/single_ap_traces:.1f}%)")
    print(f"  Failed propagation: {failed_propagation:,} ({100*failed_propagation/single_ap_traces:.1f}%)")
    print("="*80 + "\n")

    # Create summary dictionary
    summary = {
        'total_traces': total_traces,
        'single_ap_traces': single_ap_traces,
        'successful_propagation': successful_propagation,
        'failed_propagation': failed_propagation,
        'success_rate': 100.0 * successful_propagation / single_ap_traces if single_ap_traces > 0 else 0.0,
        'failure_rate': 100.0 * failed_propagation / single_ap_traces if single_ap_traces > 0 else 0.0
    }

    return {
        'amplitudes_proximal': amplitudes_proximal,
        'amplitudes_distal': amplitudes_distal,
        'speeds': propagation_speeds,
        'summary': summary
    }


def compute_statistics_from_disk(save_path, num_chunks, time_test):
    """
    Compute statistics (mean, std) from chunk files or transposed files without loading all data.
    Handles NaN values by replacing them with previous valid samples.

    Parameters:
    -----------
    save_path : str
        Directory containing chunk files or transposed files
    num_chunks : int
        Number of chunk files
    time_test : array
        Time vector

    Returns:
    --------
    mean_V, std_V : numpy arrays
        Mean and standard deviation of voltage traces
    """
    import glob

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

        for ti in range(nT):
            if ti % 2000 == 0:
                print(f"  Progress: {ti}/{nT} ({100*ti/nT:.1f}%)")

            # Collect data for this time point from all transposed chunks
            all_V_ti = []
            for f in transposed_files:
                chunk = np.load(f, mmap_mode='r')
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
                print(f"Warning: Could not read {chunk_file}: {e}")

        print(f"Computing statistics from {total_samples} samples across {len(chunk_files)} chunk files...")

        # Compute mean with NaN handling (use nanmean instead of mean)
        mean_V = np.zeros(nT)
        for chunk_file in chunk_files:
            try:
                chunk = np.load(chunk_file)
                mean_V += np.nansum(chunk, axis=0)
            except Exception as e:
                print(f"Warning: Could not read {chunk_file} for mean calculation: {e}")
        mean_V /= total_samples

        # Compute standard deviation with NaN handling
        var_V = np.zeros(nT)
        for chunk_file in chunk_files:
            try:
                chunk = np.load(chunk_file)
                var_V += np.nansum((chunk - mean_V)**2, axis=0)
            except Exception as e:
                print(f"Warning: Could not read {chunk_file} for variance calculation: {e}")
        std_V = np.sqrt(var_V / total_samples)

    return mean_V, std_V


def save_sample_traces(save_path, num_chunks, output_file='all_V_sample.csv', num_samples=1000):
    """
    Save a sample of voltage traces from chunk files.

    Parameters:
    -----------
    save_path : str
        Directory containing chunk files
    num_chunks : int
        Number of chunk files
    output_file : str
        Output CSV file name
    num_samples : int
        Number of samples to save
    """
    print(f"Saving {num_samples} sample traces...")

    # Find all injection chunk files (including sub-chunks)
    chunk_files = []
    for chunk_idx in range(num_chunks):
        main_chunk_file = os.path.join(save_path, f'chunk_{chunk_idx}_injection.npy')
        if os.path.exists(main_chunk_file):
            chunk_files.append(main_chunk_file)
        else:
            # Look for sub-chunks
            sub_idx = 0
            while True:
                sub_chunk_file = os.path.join(save_path, f'chunk_{chunk_idx}_sub_{sub_idx}_injection.npy')
                if os.path.exists(sub_chunk_file):
                    chunk_files.append(sub_chunk_file)
                    sub_idx += 1
                else:
                    break

    if not chunk_files:
        print("No chunk files found for sampling")
        return

    # Load first chunk to get dimensions
    first_chunk = np.load(chunk_files[0])
    nT = first_chunk.shape[1]

    # Collect samples
    samples_collected = 0
    sample_data = []

    for chunk_file in chunk_files:
        if samples_collected >= num_samples:
            break

        try:
            chunk = np.load(chunk_file)
            samples_to_take = min(num_samples - samples_collected, chunk.shape[0])
            sample_data.append(chunk[:samples_to_take, :])
            samples_collected += samples_to_take
        except Exception as e:
            print(f"Warning: Could not read {chunk_file} for sampling: {e}")

    if sample_data:
        # Combine and save
        all_samples = np.vstack(sample_data)
        np.savetxt(output_file, all_samples, delimiter=',', header=f'{nT} time points, {samples_collected} samples', comments='')
        print(f"Saved {samples_collected} sample traces to {output_file}")
    
    return all_samples
