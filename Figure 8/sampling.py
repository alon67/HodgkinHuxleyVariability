# sampling.py
"""
SALib/Saltelli sampling utilities and reproducibility helpers.
"""

import numpy as np
from SALib.sample import sobol as sobol_sample
from typing import Dict, List, Tuple, Any
import config


def build_problem_from_bootstrap(bootstrap_csv_path: str,
                               num_std: float = 2.0) -> Tuple[Dict[str, Any], List[float], List[float], List[str]]:
    """
    Build SALib problem definition from bootstrap parameter statistics.

    Parameters
    ----------
    bootstrap_csv_path : str
        Path to params_boot.csv
    num_std : float
        Number of standard deviations for parameter bounds

    Returns
    -------
    problem : dict
        SALib problem definition
    base_params : List[float]
        Parameter means
    bounds : List[Tuple[float, float]]
        Parameter bounds as (low, high) pairs
    param_names : List[str]
        Parameter names
    """
    from parameters import load_bootstrap_params, compute_parameter_bounds

    base_params, std_params, param_names = load_bootstrap_params(bootstrap_csv_path)
    low, high = compute_parameter_bounds(base_params, std_params, num_std)
    bounds = list(zip(low, high))

    problem = {
        'num_vars': len(param_names),
        'names': param_names,
        'bounds': bounds
    }

    return problem, base_params, bounds, param_names


def generate_saltelli_samples(problem: Dict[str, Any], N: int,
                            calc_second_order: bool = True) -> np.ndarray:
    """
    Generate Saltelli samples for sensitivity analysis.

    Parameters
    ----------
    problem : dict
        SALib problem definition
    N : int
        Base sample size
    calc_second_order : bool
        Whether to include second-order effects

    Returns
    -------
    param_values : np.ndarray
        Parameter samples (shape: N*(2*D+2) x D for calc_second_order=True)
    """
    print(f"Generating Saltelli samples: N={N}, calc_second_order={calc_second_order}")
    print(f"Problem has {problem['num_vars']} parameters")

    param_values = sobol_sample.sample(problem, N, calc_second_order=calc_second_order)

    expected_samples = N * (2 * problem['num_vars'] + 2) if calc_second_order else N * (problem['num_vars'] + 2)
    print(f"Generated {len(param_values)} parameter sets ({expected_samples} expected)")

    return param_values


def validate_sample_structure(param_values: np.ndarray, problem: Dict[str, Any],
                            calc_second_order: bool = True) -> bool:
    """
    Validate that samples have correct Saltelli structure.

    Parameters
    ----------
    param_values : np.ndarray
        Parameter samples
    problem : dict
        SALib problem definition
    calc_second_order : bool
        Whether second-order effects were included

    Returns
    -------
    bool
        True if structure is valid
    """
    D = problem['num_vars']
    expected_samples = len(param_values)
    expected_base = expected_samples // (2 * D + 2) if calc_second_order else expected_samples // (D + 2)

    if calc_second_order:
        required_samples = expected_base * (2 * D + 2)
    else:
        required_samples = expected_base * (D + 2)

    if expected_samples != required_samples:
        print(f"WARNING: Sample count ({expected_samples}) doesn't match Saltelli structure")
        print(f"Expected: N * {(2*D+2) if calc_second_order else (D+2)} = {required_samples}")
        return False

    print(f"✓ Sample structure validated: {expected_samples} samples, N={expected_base}")
    return True