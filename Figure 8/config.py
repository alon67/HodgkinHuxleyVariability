# config.py
"""
Configuration constants for HH model sensitivity analysis.
Centralizes all user-configurable settings without importing JAX.
"""

# Cable discretization and geometry constants
# (These are not used in the current simulation but kept for completeness)
CABLE_LENGTH = 1000.0  # μm
DX = 10.0  # μm
NUM_SEGMENTS = int(CABLE_LENGTH / DX)

# Time integration settings
TMAX = 100.0  # ms
DT = 0.005  # ms
SOLVER_TOLERANCE = 1e-6

# Stimulus timing and amplitude defaults
STIMULUS_START = 10.0  # ms
STIMULUS_END = 80.0  # ms
I_AMPLITUDE_DEFAULT = 30.0  # μA (total current, converted to μA/cm² internally)
MEMBRANE_AREA_CM2 = 1.0  # cm² (standard HH model assumption)

# Chunking defaults for memory management
CHUNK_SIZE_DEFAULT = 12500  # Simulations per chunk
SAVE_PATH_DEFAULT = 'temp_hh_results'
FILTERED_SAVE_PATH_DEFAULT = 'temp_filtered_results'

# AP detection and categorization thresholds
AP_THRESHOLD = -20.0  # mV
VOLTAGE_CHECK_TIME = 70.0  # ms after stimulus onset
VOLTAGE_CHECK_THRESHOLD = -40.0  # mV

# Saltelli sampling defaults
N_DEFAULT = 8192  # Base sample size
CALC_SECOND_ORDER = True

# Sobol analysis settings
BOOTSTRAP_RESAMPLES_DEFAULT = 100
CONF_LEVEL = 0.95

# File paths and names (relative to working directory)
PARAMS_BOOT_CSV = 'params_boot.csv'
TIME_VECTOR_CSV = 'time_vector.csv'
PARAM_NAMES_TXT = 'param_names.txt'
MEAN_V_CSV = 'mean_V.csv'
STD_V_CSV = 'std_V.csv'
S1_TIME_CSV = 'S1_time.csv'
ST_TIME_CSV = 'ST_time.csv'
S1_TIME_WITH_HEADERS_CSV = 'S1_time_with_headers.csv'
ST_TIME_WITH_HEADERS_CSV = 'ST_time_with_headers.csv'
ALL_V_SAMPLE_CSV = 'all_V_sample.csv'
ALL_V_SAMPLE_TRANSPOSED_CSV = 'all_V_sample_transposed.csv'

# Figure output names
VOLTAGE_TRACES_PNG = 'voltage_traces.png'
VOLTAGE_TRACES_PDF = 'voltage_traces.pdf'
VOLTAGE_TRACES_WITH_STIMULUS_PNG = 'voltage_traces_with_stimulus.png'
VOLTAGE_TRACES_WITH_STIMULUS_PDF = 'voltage_traces_with_stimulus.pdf'
SOBOL_S1_PNG = 'sobol_S1.png'
SOBOL_S1_PDF = 'sobol_S1.pdf'
SOBOL_ST_PNG = 'sobol_ST.png'
SOBOL_ST_PDF = 'sobol_ST.pdf'
SOBOL_S1_70MS_PNG = 'sobol_S1_70ms.png'
SOBOL_S1_70MS_PDF = 'sobol_S1_70ms.pdf'
SOBOL_ST_70MS_PNG = 'sobol_ST_70ms.png'
SOBOL_ST_70MS_PDF = 'sobol_ST_70ms.pdf'
SOBOL_COMBINED_A4_PNG = 'sobol_combined_A4.png'
SOBOL_COMBINED_A4_PDF = 'sobol_combined_A4.pdf'


def validate_config():
    """
    Validate configuration parameters for consistency and safety.

    Raises:
    -------
    ValueError
        If any configuration parameters are invalid
    """
    # Time validation
    if TMAX <= 0:
        raise ValueError(f"TMAX must be positive, got {TMAX}")
    if DT <= 0:
        raise ValueError(f"DT must be positive, got {DT}")
    if TMAX / DT > 1e7:
        raise ValueError("Time integration would create too many time points (>10M)")

    # Stimulus validation
    if STIMULUS_START >= STIMULUS_END:
        raise ValueError(f"STIMULUS_START ({STIMULUS_START}) must be < STIMULUS_END ({STIMULUS_END})")
    if STIMULUS_START < 0 or STIMULUS_END > TMAX:
        raise ValueError("Stimulus period must be within [0, TMAX]")

    # Chunking validation
    if CHUNK_SIZE_DEFAULT <= 0:
        raise ValueError(f"CHUNK_SIZE_DEFAULT must be positive, got {CHUNK_SIZE_DEFAULT}")

    # Cable geometry validation (if used)
    if NUM_SEGMENTS <= 0:
        raise ValueError(f"NUM_SEGMENTS must be positive, got {NUM_SEGMENTS}")

    # Sampling validation
    if N_DEFAULT <= 0:
        raise ValueError(f"N_DEFAULT must be positive, got {N_DEFAULT}")

    print("✓ Configuration validation passed")


def print_config():
    """
    Print current configuration settings for reproducibility.
    """
    print("\n" + "="*60)
    print("CONFIGURATION SETTINGS")
    print("="*60)

    print(f"Time integration: TMAX={TMAX} ms, DT={DT} ms")
    print(f"Stimulus: {STIMULUS_START}-{STIMULUS_END} ms, I={I_AMPLITUDE_DEFAULT} μA")
    print(f"Chunking: size={CHUNK_SIZE_DEFAULT}, paths='{SAVE_PATH_DEFAULT}'/'{FILTERED_SAVE_PATH_DEFAULT}'")
    print(f"Sampling: N={N_DEFAULT}, second_order={CALC_SECOND_ORDER}")
    print(f"AP detection: threshold={AP_THRESHOLD} mV, check_time={VOLTAGE_CHECK_TIME} ms")
    print(f"Sobol analysis: resamples={BOOTSTRAP_RESAMPLES_DEFAULT}, conf_level={CONF_LEVEL}")

    print("="*60 + "\n")