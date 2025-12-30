"""
Parameter management module for the Hodgkin-Huxley cable model.

Handles parameter names, LaTeX formatting, bootstrap statistics loading,
and Monte Carlo parameter sampling.
"""

import numpy as np
from scipy.stats import truncnorm
import os


# Parameter names in the order used by the model
param_names = [
    # Na channel parameters
    'gbar_Na', 'E_Na', 'A_am', 'B_am', 'C_am', 'A_bm', 'D_bm',
    'A_ah', 'D_ah', 'E_bh', 'F_bh', 'G_bh',
    # K channel parameters
    'gbar_K', 'E_K', 'A_alpha', 'V_alpha', 'k_alpha', 'A_beta', 'tau_beta',
    # Additional parameters
    'C_m', 'E_l', 'G_l'
]

# LaTeX parameter name mappings for publication-quality figures
param_latex_names = {
    'gbar_Na': r'$\bar{g}_\mathrm{Na}$',
    'E_Na': r'$E_\mathrm{Na}$',
    'A_am': r'$A_{\alpha_m}$',
    'B_am': r'$V1/2_{\alpha_m}$',
    'C_am': r'$z_{\alpha_m}$',
    'A_bm': r'$A_{\beta_m}$',
    'D_bm': r'$z_{\beta_m}$',
    'A_ah': r'$A_{\alpha_h}$',
    'D_ah': r'$z_{\alpha_h}$',
    'E_bh': r'$E_{\beta_h}$',
    'F_bh': r'$V1/2_{\beta_h}$',
    'G_bh': r'$z_{\beta_h}$',
    'gbar_K': r'$\bar{g}_\mathrm{K}$',
    'E_K': r'$E_\mathrm{K}$',
    'A_alpha': r'$A_{\alpha_n}$',
    'V_alpha': r'$V1/2_{\alpha_n}$',
    'k_alpha': r'$z_{\alpha_n}$',
    'A_beta': r'$A_{\beta_n}$',
    'tau_beta': r'$z_{\beta_n}$',
    'C_m': r'$C_\mathrm{m}$',
    'E_l': r'$E_\mathrm{L}$',
    'G_l': r'$\bar{g}_\mathrm{L}$'
}


def get_latex_name(param_name):
    """
    Get LaTeX formatted name for parameter, fall back to original name if not found.

    Parameters:
    -----------
    param_name : str
        Parameter name

    Returns:
    --------
    str
        LaTeX formatted parameter name
    """
    return param_latex_names.get(param_name, param_name)


def load_bootstrap_stats(path):
    """
    Load bootstrap statistics from CSV file.

    Parameters:
    -----------
    path : str
        Path to the bootstrap CSV file

    Returns:
    --------
    list of tuple
        List of (mean, std) pairs for each parameter
    """
    bootstrap_data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Parse mean and std from format "mean, std # comment"
                parts = line.split('#')[0].strip().split(',')
                if len(parts) >= 2:
                    mean = float(parts[0].strip())
                    std = float(parts[1].strip())
                    bootstrap_data.append((mean, std))

    if len(bootstrap_data) != len(param_names):
        raise ValueError(f"Bootstrap data has {len(bootstrap_data)} entries but expected {len(param_names)} parameters")

    return bootstrap_data


def load_or_generate_param_values(bootstrap_file, param_file, N, seed=None):
    """
    Load existing parameter values or generate new ones using truncated normal sampling.

    Parameters:
    -----------
    bootstrap_file : str
        Path to bootstrap statistics file
    param_file : str
        Path to parameter values file
    N : int
        Number of parameter sets to generate
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    numpy.ndarray
        Parameter values array of shape (N, n_params)
    """
    # Try to load existing parameter values
    if os.path.exists(param_file):
        print(f"Loading existing parameter values from {param_file}")
        param_values = np.loadtxt(param_file, delimiter=',', skiprows=1)  # Skip header

        if param_values.shape[1] != len(param_names):
            raise ValueError(f"Parameter dimension mismatch: {param_values.shape[1]} vs {len(param_names)}")

        if param_values.shape[0] != N:
            print(f"Warning: Loaded {param_values.shape[0]} parameter sets but requested {N}")
            if param_values.shape[0] < N:
                print("Generating additional parameter sets...")
                # Generate additional sets
                additional_values = _generate_param_values(bootstrap_file, N - param_values.shape[0], seed)
                param_values = np.vstack([param_values, additional_values])
            else:
                print(f"Using first {N} parameter sets from loaded data")
                param_values = param_values[:N]

        print(f"✓ Loaded parameter values: {param_values.shape}")
        return param_values

    # Generate new parameter values
    print(f"Generating {N} parameter sets using truncated normal sampling")
    param_values = _generate_param_values(bootstrap_file, N, seed)

    # Save to file
    print(f"Saving parameter values to {param_file}")
    header = ','.join(param_names)
    np.savetxt(param_file, param_values, delimiter=',', header=header, comments='')

    print(f"✓ Generated and saved parameter values: {param_values.shape}")
    return param_values


def _generate_param_values(bootstrap_file, N, seed=None):
    """
    Generate parameter values using truncated normal sampling.

    Parameters:
    -----------
    bootstrap_file : str
        Path to bootstrap statistics file
    N : int
        Number of parameter sets to generate
    seed : int, optional
        Random seed

    Returns:
    --------
    numpy.ndarray
        Parameter values array of shape (N, n_params)
    """
    if seed is not None:
        np.random.seed(seed)

    # Load bootstrap statistics
    bootstrap_data = load_bootstrap_stats(bootstrap_file)

    param_values = np.zeros((N, len(param_names)))

    for i, (param_name, (mean, std)) in enumerate(zip(param_names, bootstrap_data)):
        # Define truncation bounds: [mean - 2*std, mean + 2*std]
        lower_bound = mean - 2 * std
        upper_bound = mean + 2 * std

        # For parameters with positive mean, enforce minimum bound of 1e-10
        # to avoid negative conductances/scale parameters
        if mean > 0:
            lower_bound = max(lower_bound, 1e-10)

        # For parameters with negative mean (e.g., reversal potentials), allow negatives
        # No additional constraints needed

        # Convert to scipy truncnorm parameters
        # truncnorm takes a, b in standard normal units
        a = (lower_bound - mean) / std  # Lower bound in std units
        b = (upper_bound - mean) / std  # Upper bound in std units

        # Sample from truncated normal
        samples = truncnorm.rvs(a, b, loc=mean, scale=std, size=N)

        # Store in parameter array
        param_values[:, i] = samples

    return param_values


def compute_param_bounds(param_values):
    """
    Compute parameter bounds for plotting.

    Parameters:
    -----------
    param_values : numpy.ndarray
        Parameter values array

    Returns:
    --------
    list of tuple
        List of (min, max) bounds for each parameter
    """
    bounds = []
    for i in range(param_values.shape[1]):
        param_data = param_values[:, i]
        # Use 1st and 99th percentiles for bounds to avoid outliers
        lower = np.percentile(param_data, 1)
        upper = np.percentile(param_data, 99)
        bounds.append((lower, upper))
    return bounds
