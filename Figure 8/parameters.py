# parameters.py
"""
Parameter definitions, LaTeX names, and bootstrap I/O utilities.
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple

# Parameter names in the exact order used by the simulation
PARAM_NAMES = [
    # Na channel parameters
    'gbar_Na', 'E_Na', 'A_am', 'B_am', 'C_am', 'A_bm', 'D_bm', 'A_ah', 'D_ah', 'E_bh', 'F_bh', 'G_bh',
    # K channel parameters
    'gbar_K', 'E_K', 'A_alpha', 'V_alpha', 'k_alpha', 'A_beta', 'tau_beta',
    # Additional parameters
    'C_m', 'E_l', 'G_l'
]

# LaTeX formatted names for publication-quality figures
PARAM_LATEX_NAMES = {
    # Na channel parameters
    'gbar_Na': r'$\bar{g}_\mathrm{Na}$',
    'E_Na': r'$E_\mathrm{Na}$',
    'A_am': r'$A_{\alpha,m}$',
    'B_am': r'$V1/2_{\alpha,m}$',
    'C_am': r'$z_{\alpha,m}$',
    'A_bm': r'$A_{\beta,m}$',
    'D_bm': r'$z_{\beta,m}$',
    'A_ah': r'$A_{\alpha,h}$',
    'D_ah': r'$z_{\alpha,h}$',
    'E_bh': r'$A_{\beta,h}$',
    'F_bh': r'$V1/2_{\beta,h}$',
    'G_bh': r'$z_{\beta,h}$',
    # K channel parameters
    'gbar_K': r'$\bar{g}_\mathrm{K}$',
    'E_K': r'$E_\mathrm{K}$',
    'A_alpha': r'$A_{\alpha,n}$',
    'V_alpha': r'$V1/2_{\alpha,n}$',
    'k_alpha': r'$z_{\alpha,n}$',
    'A_beta': r'$A_{\beta,n}$',
    'tau_beta': r'$z_{\beta,n}$',
    # Additional parameters
    'C_m': r'$C_\mathrm{m}$',
    'E_l': r'$E_\mathrm{L}$',
    'G_l': r'$\bar{g}_\mathrm{L}$'
}


def get_latex_name(param_name: str) -> str:
    """
    Get LaTeX formatted name for a parameter, fallback to code name if not mapped.

    Parameters
    ----------
    param_name : str
        Parameter name

    Returns
    -------
    str
        LaTeX formatted name
    """
    return PARAM_LATEX_NAMES.get(param_name, f'${param_name}$')


def load_bootstrap_params(bootstrap_csv_path: str) -> Tuple[List[float], List[float], List[str]]:
    """
    Load parameter means and standard deviations from bootstrap CSV file.

    Parameters
    ----------
    bootstrap_csv_path : str
        Path to params_boot.csv file

    Returns
    -------
    base_params : List[float]
        Parameter means
    std_params : List[float]
        Parameter standard deviations
    param_names : List[str]
        Parameter names (should match PARAM_NAMES)
    """
    params_data = []
    with open(bootstrap_csv_path, 'r') as f:
        for line in f:
            # Parse each line: "mean, std # param_name"
            parts = line.split('#')[0].strip()  # Remove comment
            values = parts.split(',')
            mean_val = float(values[0].strip())
            std_val = float(values[1].strip())
            params_data.append((mean_val, std_val))

    base_params = [mean for mean, std in params_data]
    std_params = [std for mean, std in params_data]

    return base_params, std_params, PARAM_NAMES


def compute_parameter_bounds(base_params: List[float], std_params: List[float],
                           num_std: float = 2.0) -> Tuple[List[float], List[float]]:
    """
    Compute parameter bounds as mean +/- num_std * std.

    Parameters
    ----------
    base_params : List[float]
        Parameter means
    std_params : List[float]
        Parameter standard deviations
    num_std : float
        Number of standard deviations for bounds

    Returns
    -------
    low : List[float]
        Lower bounds
    high : List[float]
        Upper bounds
    """
    low = []
    high = []
    for mean_val, std_val in zip(base_params, std_params):
        lower = mean_val - num_std * std_val
        upper = mean_val + num_std * std_val

        # For parameters with positive mean, clip lower bound at 1e-10
        if mean_val > 0:
            lower = max(lower, 1e-10)

        low.append(lower)
        high.append(upper)

    return low, high


def save_param_names_txt(param_names: List[str], output_path: str) -> None:
    """
    Save parameter names to text file (one per line).

    Parameters
    ----------
    param_names : List[str]
        Parameter names
    output_path : str
        Output file path
    """
    with open(output_path, 'w') as f:
        for name in param_names:
            f.write(f"{name}\n")


def load_param_names_txt(param_names_path: str) -> List[str]:
    """
    Load parameter names from text file.

    Parameters
    ----------
    param_names_path : str
        Path to param_names.txt

    Returns
    -------
    List[str]
        Parameter names
    """
    with open(param_names_path, 'r') as f:
        return [line.strip() for line in f]