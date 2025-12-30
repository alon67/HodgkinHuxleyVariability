# Hodgkin-Huxley Cable Model Monte Carlo Analysis

A modular, JAX-accelerated implementation for Monte Carlo analysis of action potential propagation in a Hodgkin-Huxley cable model.

## Overview

This package performs comprehensive Monte Carlo simulations of the Hodgkin-Huxley cable model to analyze action potential propagation patterns under parameter uncertainty. The implementation uses JAX for high-performance, JIT-compiled simulations and provides detailed analysis of firing patterns, propagation success rates, and parameter sensitivities.

## Features

- **JAX-accelerated simulations**: High-performance Monte Carlo simulations using JAX and diffrax
- **Modular architecture**: Clean separation of configuration, simulation, analysis, and visualization
- **Comprehensive analysis**: Automatic categorization of voltage traces, propagation analysis, and statistical summaries
- **Publication-quality figures**: Automated generation of histograms, scatter plots, and trace overlays
- **Parameter sensitivity**: Analysis of how model parameters affect firing patterns and propagation

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Run Monte Carlo analysis with 10,000 samples:
```bash
python main.py --n-samples 10000
```

### Custom Save Path

```bash
python main.py --n-samples 5000 --save-path ./my_results
```

### Analyze Existing Data

```bash
python main.py --analyze-only --save-path ./existing_simulation_data
```

### View Configuration

```bash
python main.py --config-only
```

## Module Structure

- **`config.py`**: Centralized configuration management and validation
- **`parameters.py`**: Parameter loading, sampling, and LaTeX formatting
- **`core_simulation.py`**: JAX-based Hodgkin-Huxley simulation functions
- **`core_analysis.py`**: Analysis functions for trace categorization and statistics
- **`graphics.py`**: Plotting functions for publication-quality figures
- **`main.py`**: Command-line interface and workflow orchestration

## Key Parameters

- **Cable geometry**: 10 cm length, 0.4 cm injection site, 8.0 cm recording site
- **Stimulus**: 70 ms current injection (amplitude configurable)
- **Parameters**: 22 Hodgkin-Huxley parameters sampled from bootstrap distributions
- **Analysis**: Automatic categorization into 5 firing pattern categories

## Output Files

### Figures (PNG/PDF)
- `trace_categories_pie.png`: Distribution of firing patterns
- `representative_traces.png`: Example traces for each category
- `parameter_distributions_*.png`: Parameter distributions by category
- `ap_amplitude_*_histogram.png`: Action potential amplitude distributions
- `propagation_speed_histogram.png`: Conduction velocity distribution

### Data Files (CSV)
- `resting_potentials.csv`: Resting membrane potentials
- `firing_frequencies_raw.csv`: Firing frequencies for repetitive firing
- `ap_amplitudes_*.csv`: Action potential amplitudes
- `propagation_speeds.csv`: Propagation speeds

### Summary Files
- `analysis_summary_report.txt`: Comprehensive analysis summary
- `propagation_analysis_summary.txt`: Detailed propagation statistics

## Analysis Categories

The software automatically categorizes voltage traces into 5 patterns:

1. **No Action Potential**: No spikes during entire simulation
2. **Single AP - Propagated**: One spike after stimulus onset, successfully propagates
3. **Single AP - Failed**: One spike after stimulus onset, fails to propagate
4. **Multiple APs During Stimulus**: Sustained repetitive firing
5. **AP Before Stimulus**: Spontaneous activity before stimulus

## Technical Details

- **Backend**: JAX with CPU backend (64-bit precision)
- **ODE Solver**: diffrax with adaptive time stepping
- **Parallelization**: JAX pmap/vmap for multi-core execution
- **Memory Management**: Chunked processing for large sample sizes
- **Parameter Sampling**: Truncated normal distributions with positivity constraints

## Requirements

- Python 3.8+
- JAX 0.4.0+
- NumPy, SciPy, Matplotlib
- diffrax (ODE solver)

## Citation

If you use this code in your research, please cite the original Hodgkin-Huxley model and acknowledge the JAX implementation for performance improvements.

## License

This software is provided as-is for research purposes.