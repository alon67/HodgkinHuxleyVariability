# sim_io.py
"""
Chunking, saving/loading, and transposition utilities for simulation data.
Handles disk I/O for large simulation datasets.
"""

import numpy as np
import os
import time
import gc
from typing import List, Tuple, Optional
import config
import sim_core


def incremental_simulation_jax(param_values: np.ndarray, time_test: np.ndarray, dt: float, nT: int,
                             chunk_size: int = config.CHUNK_SIZE_DEFAULT,
                             save_path: str = config.SAVE_PATH_DEFAULT,
                             I_amplitude: float = config.I_AMPLITUDE_DEFAULT) -> Tuple[str, int]:
    """
    Run simulations in chunks using JAX vectorization and save results incrementally to disk.
    Much faster than multiprocessing approach due to JIT compilation and vectorization.

    Parameters
    ----------
    param_values : np.ndarray
        Parameter sets from sampling (NumPy array)
    time_test : np.ndarray
        Time vector
    dt : float
        Time step
    nT : int
        Number of time points
    chunk_size : int
        Number of simulations to run before saving
    save_path : str
        Directory to save temporary results
    I_amplitude : float
        Current injection amplitude in μA (converted to μA/cm² internally)

    Returns
    -------
    save_path : str
        Path to directory containing chunk files
    num_chunks : int
        Number of chunk files created
    """
    # Create temporary directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    total_sims = len(param_values)
    num_chunks = int(np.ceil(total_sims / chunk_size))

    print(f"Running JAX simulations in {num_chunks} chunks of {chunk_size}...")
    print(f"Expected massive speedup from JIT compilation + vectorization!")
    print(f"\nNOTE: First chunk will be slower due to JIT compilation (~20-40 seconds).")
    print(f"      Subsequent chunks will be much faster (~3-8 seconds each).\n")

    # Pre-compute static values to avoid recompilation
    tmax_static = float(time_test[-1])
    dt_static = float(dt)

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min((chunk_idx + 1) * chunk_size, total_sims)
        actual_chunk_size = chunk_end - chunk_start

        print(f"\nProcessing JAX chunk {chunk_idx + 1}/{num_chunks} (samples {chunk_start}-{chunk_end})...")
        chunk_start_time = time.time()

        # Get parameter subset for this chunk
        params_chunk = param_values[chunk_start:chunk_end]

        # Pad last chunk to match chunk_size to avoid recompilation
        if actual_chunk_size < chunk_size:
            # Repeat last parameter set to fill chunk
            padding_size = chunk_size - actual_chunk_size
            padding = np.tile(params_chunk[-1:], (padding_size, 1))
            params_chunk = np.vstack([params_chunk, padding])
            print(f"  Note: Padded last chunk from {actual_chunk_size} to {chunk_size} samples")

        # Convert to JAX array
        params_jax = sim_core.jnp.array(params_chunk)

        try:
            # Run vectorized simulation - THIS IS WHERE THE MAGIC HAPPENS!
            print(f"  Running {chunk_size} simulations in parallel (I={I_amplitude} μA)...")
            sim_start = time.time()
            _, voltage_traces = sim_core.simulate_batch_jax(params_jax, tmax_static, dt_static, I_amplitude)
            sim_time = time.time() - sim_start

            # Trim padding from results if needed
            if actual_chunk_size < chunk_size:
                voltage_traces = voltage_traces[:actual_chunk_size]

            # Convert back to NumPy for saving
            chunk_V = np.array(voltage_traces, dtype=np.float32)

            print(f"  Simulations completed in {sim_time:.2f}s ({actual_chunk_size/sim_time:.1f} sims/sec)")

            # Save chunk to disk
            chunk_file = os.path.join(save_path, f'chunk_{chunk_idx}.npy')
            estimated_size_mb = chunk_V.nbytes / (1024**2)
            print(f"  Saving chunk {chunk_idx + 1}/{num_chunks} (~{estimated_size_mb:.1f} MB)...")

            np.save(chunk_file, chunk_V)

            # Verify save
            if os.path.exists(chunk_file) and os.path.getsize(chunk_file) > 1000:
                chunk_total_time = time.time() - chunk_start_time
                print(f"  ✓ Chunk {chunk_idx + 1}/{num_chunks} saved successfully (total: {chunk_total_time:.2f}s)")
            else:
                print(f"  WARNING: Chunk {chunk_idx + 1} may not have saved correctly")

        except Exception as e:
            print(f"  ERROR during chunk {chunk_idx + 1}: {e}")
            print(f"  Attempting to save partial results...")

            # Try saving smaller sub-batches
            try:
                sub_batch_size = min(1000, actual_chunk_size // 4)
                for sub_idx in range(0, actual_chunk_size, sub_batch_size):
                    sub_end = min(sub_idx + sub_batch_size, actual_chunk_size)
                    sub_params = params_jax[sub_idx:sub_end]
                    _, sub_traces = sim_core.simulate_batch_jax(sub_params, tmax_static, dt_static, I_amplitude)
                    sub_chunk = np.array(sub_traces, dtype=np.float32)
                    sub_file = os.path.join(save_path, f'chunk_{chunk_idx}_sub_{sub_idx//sub_batch_size}.npy')
                    np.save(sub_file, sub_chunk)
                print(f"  Chunk {chunk_idx + 1} saved as sub-chunks")
            except Exception as e2:
                print(f"  CRITICAL ERROR: Could not save chunk {chunk_idx + 1}: {e2}")
                raise

        # Free memory
        del chunk_V

    # Clean up JAX resources after all simulations complete
    print("\nCleaning up JAX resources...")
    sim_core.jax.clear_caches()  # Clear JIT compilation cache
    gc.collect()  # Force Python garbage collection
    print("JAX cleanup complete. Memory released for Sobol analysis.")

    return save_path, num_chunks


def transpose_chunks_for_sobol(save_path: str, num_chunks: int, nT: int) -> None:
    """
    Transpose chunk files from (samples × time) to (time × samples) format.
    This is a one-time preprocessing step that dramatically speeds up Sobol analysis.

    After transposition:
    - Original files: chunk_N.npy (samples × time)
    - Transposed files: chunk_N_transposed.npy (time × samples)

    Parameters
    ----------
    save_path : str
        Directory containing chunk files
    num_chunks : int
        Expected number of chunk files (used as a guide, but will auto-detect all files)
    nT : int
        Number of time points
    """
    print("\n" + "="*80)
    print("TRANSPOSING DATA FOR EFFICIENT SOBOL ANALYSIS")
    print("="*80)
    print("This one-time operation will dramatically speed up Sobol analysis...\n")

    # AUTO-DETECT all chunk files (both main and sub-chunks)
    # This is more robust than relying on num_chunks parameter
    chunk_files = []
    chunk_identifiers = []  # Store (chunk_idx, sub_idx or None)

    # Scan directory for ALL chunk files
    import glob
    all_files = glob.glob(os.path.join(save_path, 'chunk_*.npy'))

    for file_path in sorted(all_files):
        filename = os.path.basename(file_path)
        # Skip already transposed files
        if '_transposed' in filename:
            continue

        # Parse chunk_N.npy or chunk_N_sub_M.npy
        if '_sub_' in filename:
            # Sub-chunk: chunk_N_sub_M.npy
            parts = filename.replace('.npy', '').split('_')
            chunk_idx = int(parts[1])
            sub_idx = int(parts[3])
            chunk_identifiers.append((chunk_idx, sub_idx))
        else:
            # Main chunk: chunk_N.npy
            chunk_idx = int(filename.replace('chunk_', '').replace('.npy', ''))
            chunk_identifiers.append((chunk_idx, None))

        chunk_files.append(file_path)

    print(f"Auto-detected {len(chunk_files)} chunk files to transpose")
    if len(chunk_files) == 0:
        print("ERROR: No chunk files found!")
        return

    # Transpose each chunk file
    for idx, (chunk_file, (chunk_idx, sub_idx)) in enumerate(zip(chunk_files, chunk_identifiers)):
        try:
            # Check if already transposed
            if sub_idx is None:
                transposed_file = os.path.join(save_path, f'chunk_{chunk_idx}_transposed.npy')
            else:
                transposed_file = os.path.join(save_path, f'chunk_{chunk_idx}_sub_{sub_idx}_transposed.npy')

            if os.path.exists(transposed_file):
                print(f"  [{idx+1}/{len(chunk_files)}] Already transposed: {os.path.basename(transposed_file)}")
                continue

            # Load original chunk (samples × time)
            print(f"  [{idx+1}/{len(chunk_files)}] Transposing {os.path.basename(chunk_file)}...", end=' ')
            chunk = np.load(chunk_file)

            # Transpose to (time × samples)
            chunk_transposed = chunk.T.astype(np.float32)  # Ensure float32 for memory efficiency

            # Save transposed version
            np.save(transposed_file, chunk_transposed)

            file_size_mb = os.path.getsize(transposed_file) / (1024**2)
            print(f"✓ ({file_size_mb:.1f} MB)")

            # Free memory
            del chunk, chunk_transposed

            # Delete original chunk to free disk space immediately
            os.remove(chunk_file)
            print(f"  → Deleted original chunk to free space")

        except Exception as e:
            print(f"  ✗ ERROR transposing {chunk_file}: {e}")
            raise

    print("\n" + "="*80)
    print("TRANSPOSITION COMPLETE")
    print("="*80 + "\n")


def list_chunk_files(save_path: str) -> List[str]:
    """
    List all chunk files in a directory.

    Parameters
    ----------
    save_path : str
        Directory containing chunk files

    Returns
    -------
    List[str]
        List of chunk file paths
    """
    import glob
    return sorted(glob.glob(os.path.join(save_path, 'chunk_*.npy')))


def infer_num_chunks(save_path: str) -> int:
    """
    Infer the number of main chunks from existing files.

    Parameters
    ----------
    save_path : str
        Directory containing chunk files

    Returns
    -------
    int
        Number of main chunks
    """
    chunk_files = list_chunk_files(save_path)
    if not chunk_files:
        return 0

    # Find maximum chunk index
    max_idx = -1
    for f in chunk_files:
        basename = os.path.basename(f)
        if '_sub_' not in basename and '_transposed' not in basename:
            idx = int(basename.replace('chunk_', '').replace('.npy', ''))
            max_idx = max(max_idx, idx)

    return max_idx + 1 if max_idx >= 0 else 0


def load_chunk(save_path: str, idx: int) -> np.ndarray:
    """
    Load a specific chunk file.

    Parameters
    ----------
    save_path : str
        Directory containing chunk files
    idx : int
        Chunk index

    Returns
    -------
    np.ndarray
        Chunk data
    """
    chunk_file = os.path.join(save_path, f'chunk_{idx}.npy')
    return np.load(chunk_file)