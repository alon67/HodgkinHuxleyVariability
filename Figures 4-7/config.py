"""
Configuration module for the JAX-optimized spatial Hodgkin-Huxley cable model.

This module contains all user-configurable parameters and provides validation
and utility functions for the simulation setup.
"""

import os
import numpy as np


# Cable geometry and discretization
L_AXON = 10.0  # Length of axon in cm
A_RADIUS = 0.025  # Radius of axon in cm (0.5 mm diameter / 2)
R_AXOPLASM = 35.4  # Specific axoplasmic resistance in ohm*cm
N_SEGMENTS = 80  # Number of spatial segments

# Spatial discretization
DX = L_AXON / N_SEGMENTS  # Spatial step in cm

# Injection and recording locations
INJECTION_DISTANCE = 0.4  # cm from left end
INJECTION_SEGMENT = int(INJECTION_DISTANCE / DX)  # Segment index for injection
RECORDING_DISTANCE = 8.0  # cm from left end
RECORDING_SEGMENT = int(RECORDING_DISTANCE / DX)  # Segment index for distal recording

# Cable coupling coefficient
LAMBDA_CABLE = (np.pi * A_RADIUS**2) / R_AXOPLASM  # S*cm

# Adaptive timestep for numerical stability
# For stability: dt < C_m * dx² / (2 * D_eff) where D_eff = LAMBDA/(2πa) * 1000
# With C_m ≈ 1.0 μF/cm², use safety factor of 0.4 for 2nd-order Heun method
LAMBDA_CABLE_PRELIM = (np.pi * A_RADIUS**2) / R_AXOPLASM  # S*cm
D_EFF = LAMBDA_CABLE_PRELIM / (2.0 * np.pi * A_RADIUS) * 1000.0
DT_STABLE = 0.4 * DX**2 * 1.0 / (2.0 * D_EFF)  # Safety factor 0.4 for Heun method
DT = float(np.round(DT_STABLE, 6))  # Round to avoid floating point issues

# Stimulus timing parameters
T_START = 10.0  # Stimulus start time (ms)
T_END = 80.0    # Stimulus end time (ms)
RAMP_T = 1.0    # Ramp time (ms)

# Default simulation parameters
DEFAULT_I_AMPLITUDE = 5.0  # Total injected current (μA)
DEFAULT_TMAX = 100.0       # Simulation duration (ms)

# JAX/XLA configuration
XLA_DEVICE_COUNT = 11  # Number of CPU cores to use (leave 3 free)
XLA_INTRA_OP_THREADS = 11

# Chunking and I/O
DEFAULT_CHUNK_SIZE = 5000
DEFAULT_SAVE_PATH = 'temp_hh_results_jax'
DEFAULT_N_SAMPLES = 300000

# Bootstrap and parameter files
DEFAULT_BOOTSTRAP_FILE = 'params_boot.csv'
DEFAULT_PARAM_FILE = 'param_values.csv'


def configure_xla_environment():
    """
    Configure XLA environment variables for multi-core CPU support.
    MUST be called before importing JAX.
    """
    os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={XLA_DEVICE_COUNT}'
    os.environ['XLA_FLAGS'] += f' --xla_cpu_multi_thread_eigen=true'
    os.environ['XLA_FLAGS'] += f' intra_op_parallelism_threads={XLA_INTRA_OP_THREADS}'


def validate_configuration():
    """
    Validate configuration parameters for consistency and physical reasonableness.

    Raises:
        ValueError: If any configuration parameters are invalid.
    """
    # Check positive dimensions
    if L_AXON <= 0:
        raise ValueError(f"Axon length must be positive, got {L_AXON}")
    if A_RADIUS <= 0:
        raise ValueError(f"Axon radius must be positive, got {A_RADIUS}")
    if R_AXOPLASM <= 0:
        raise ValueError(f"Axoplasmic resistance must be positive, got {R_AXOPLASM}")
    if N_SEGMENTS <= 0:
        raise ValueError(f"Number of segments must be positive, got {N_SEGMENTS}")

    # Check segment indices are valid
    if not (0 <= INJECTION_SEGMENT < N_SEGMENTS):
        raise ValueError(f"Injection segment {INJECTION_SEGMENT} out of range [0, {N_SEGMENTS-1}]")
    if not (0 <= RECORDING_SEGMENT < N_SEGMENTS):
        raise ValueError(f"Recording segment {RECORDING_SEGMENT} out of range [0, {N_SEGMENTS-1}]")

    # Check injection is before recording
    if INJECTION_SEGMENT >= RECORDING_SEGMENT:
        raise ValueError("Injection segment must be before recording segment")

    # Check stimulus timing
    if T_START >= T_END:
        raise ValueError(f"Stimulus start time {T_START} must be before end time {T_END}")
    if RAMP_T <= 0:
        raise ValueError(f"Ramp time must be positive, got {RAMP_T}")

    # Check timestep is reasonable
    if DT <= 0:
        raise ValueError(f"Timestep must be positive, got {DT}")

    # Check XLA configuration
    if XLA_DEVICE_COUNT <= 0:
        raise ValueError(f"XLA device count must be positive, got {XLA_DEVICE_COUNT}")


def print_configuration():
    """
    Print a summary of the current configuration.
    """
    print("Spatial Cable Model Configuration")
    print("=" * 40)
    print(f"Axon length: {L_AXON} cm")
    print(f"Axon radius: {A_RADIUS} cm ({A_RADIUS*10} mm)")
    print(f"Number of segments: {N_SEGMENTS}")
    print(f"Spatial resolution (dx): {DX:.4f} cm")
    print(f"Cable coupling coefficient: {LAMBDA_CABLE:.6f} S*cm")
    print(f"Adaptive timestep: {DT:.6f} ms (stability limit: {DT_STABLE:.6f} ms)")
    print(f"Injection location: {INJECTION_DISTANCE} cm from left end (segment {INJECTION_SEGMENT})")
    print(f"Recording location: {RECORDING_DISTANCE} cm from left end (segment {RECORDING_SEGMENT})")
    print(f"Propagation distance: {RECORDING_DISTANCE - INJECTION_DISTANCE:.1f} cm")
    print(f"Stimulus timing: {T_START}-{T_END} ms with {RAMP_T} ms ramps")
    print(f"XLA configuration: {XLA_DEVICE_COUNT} CPU cores, {XLA_INTRA_OP_THREADS} threads")
    print("=" * 40)


# Validate configuration on import

# Validate configuration on import
validate_configuration()
