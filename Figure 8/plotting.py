# plotting.py
"""
Matplotlib plotting functions for Hodgkin-Huxley sensitivity analysis.
All plotting logic shared between pipeline and replot scripts.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Any, Optional
import config
import parameters


def plot_voltage_traces(data: Dict[str, Any], output_dir: str = '.') -> None:
    """Plot voltage traces with mean overlay."""
    if 'all_V_sample' not in data:
        print("  ⊗ Skipping voltage traces plot (no sample data)")
        return

    print("Plotting voltage traces...")

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot sample traces
    num_traces = min(1000, data['all_V_sample'].shape[0])
    for i in range(num_traces):
        ax.plot(data['time_test'], data['all_V_sample'][i, :], 'k-', alpha=0.15, linewidth=0.3)

    # Plot mean
    ax.plot(data['time_test'], data['mean_V'], color='#E63946', lw=3.5,
           label='Mean Voltage', zorder=10, alpha=0.95)

    ax.set_xlabel('Time (ms)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Membrane Potential (mV)', fontsize=16, fontweight='bold')
    ax.set_title('Hodgkin-Huxley Model: Single Action Potential Traces',
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xlim([0, data['time_test'][-1]])
    ax.legend(fontsize=13, frameon=False, loc='upper right')
    ax.tick_params(labelsize=13)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Enhance remaining spines
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, config.VOLTAGE_TRACES_PNG), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, config.VOLTAGE_TRACES_PDF), bbox_inches='tight')
    plt.close()
    print("  ✓ Saved voltage_traces.png and .pdf")


def plot_voltage_traces_with_stimulus(data: Dict[str, Any], output_dir: str = '.',
                                    stim_start: float = config.STIMULUS_START,
                                    stim_end: float = config.STIMULUS_END) -> None:
    """Plot voltage traces with stimulus period highlighted."""
    if 'all_V_sample' not in data:
        print("  ⊗ Skipping stimulus plot (no sample data)")
        return

    print("Plotting voltage traces with stimulus...")

    fig, ax = plt.subplots(figsize=(14, 7))

    # Highlight stimulus period
    ax.axvspan(stim_start, stim_end, alpha=0.15, color='#FFA500',
              label='Stimulus Period', zorder=0)

    # Plot sample traces
    num_traces = min(1000, data['all_V_sample'].shape[0])
    for i in range(num_traces):
        ax.plot(data['time_test'], data['all_V_sample'][i, :], 'k-', alpha=0.12, linewidth=0.3)

    # Plot mean
    ax.plot(data['time_test'], data['mean_V'], color='#E63946', lw=3.5,
           label='Mean Voltage', zorder=10, alpha=0.95)

    ax.set_xlabel('Time (ms)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Membrane Potential (mV)', fontsize=16, fontweight='bold')
    ax.set_title('Hodgkin-Huxley Model: Action Potentials with Stimulus Period',
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xlim([0, data['time_test'][-1]])
    ax.legend(fontsize=13, frameon=False, loc='upper right')
    ax.tick_params(labelsize=13)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Enhance remaining spines
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, config.VOLTAGE_TRACES_WITH_STIMULUS_PNG),
               dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, config.VOLTAGE_TRACES_WITH_STIMULUS_PDF),
               bbox_inches='tight')
    plt.close()
    print("  ✓ Saved voltage_traces_with_stimulus.png and .pdf")


def plot_sobol_S1(data: Dict[str, Any], output_dir: str = '.') -> None:
    """Plot first-order Sobol indices."""
    if 'S1_time' not in data:
        print("  ⊗ Skipping Sobol S1 plot (no data)")
        return

    print("Plotting Sobol S1 indices...")

    fig, ax = plt.subplots(figsize=(14, 7))

    num_params = len(data['param_names'])
    # Color scheme based on parameter groups
    # Na parameters (0-11): Red and hot tones
    na_colors = ['#DC143C', '#FF4500', '#FF6347', '#FF7F50', '#E63946', '#D62828',
                 '#C1121F', '#9D0208', '#FF073A', '#E01E37', '#CC2936', '#B91372']
    # K parameters (12-18): Blue and cold tones
    k_colors = ['#0077B6', '#023E8A', '#03045E', '#00B4D8', '#0096C7', '#48CAE4', '#90E0EF']
    # General parameters (19-21): Gray and B&W tones
    general_colors = ['#495057', '#6C757D', '#ADB5BD']

    param_colors = na_colors + k_colors + general_colors

    # Create a copy of S1_time for plotting
    S1_plot = data['S1_time'].copy()

    # If first value of S1 is negative, offset to zero
    for p in range(num_params):
        if S1_plot[0, p] < 0:
            S1_plot[:, p] -= S1_plot[0, p]

    for p in range(num_params):
        ax.plot(data['time_test'], S1_plot[:, p],
               label=parameters.get_latex_name(data['param_names'][p]),
               linewidth=2.5, color=param_colors[p], alpha=0.9)

    ax.set_xlabel('Time (ms)', fontsize=16, fontweight='bold')
    ax.set_ylabel('First-Order Sobol Index ($S_1$)', fontsize=16, fontweight='bold')
    ax.set_title('Time-Dependent First-Order Sensitivity Indices',
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xlim([0, data['time_test'][-1]])
    ax.set_ylim([0, 1.05])  # Slight headroom above 1
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, frameon=False)
    ax.tick_params(labelsize=13)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Enhance remaining spines
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, config.SOBOL_S1_PNG), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, config.SOBOL_S1_PDF), bbox_inches='tight')
    plt.close()
    print("  ✓ Saved sobol_S1.png and .pdf")


def plot_sobol_ST(data: Dict[str, Any], output_dir: str = '.') -> None:
    """Plot total-order Sobol indices."""
    if 'ST_time' not in data:
        print("  ⊗ Skipping Sobol ST plot (no data)")
        return

    print("Plotting Sobol ST indices...")

    fig, ax = plt.subplots(figsize=(14, 7))

    num_params = len(data['param_names'])
    # Color scheme based on parameter groups (same as S1)
    # Na parameters (0-11): Red and hot tones
    na_colors = ['#DC143C', '#FF4500', '#FF6347', '#FF7F50', '#E63946', '#D62828',
                 '#C1121F', '#9D0208', '#FF073A', '#E01E37', '#CC2936', '#B91372']
    # K parameters (12-18): Blue and cold tones
    k_colors = ['#0077B6', '#023E8A', '#03045E', '#00B4D8', '#0096C7', '#48CAE4', '#90E0EF']
    # General parameters (19-21): Gray and B&W tones
    general_colors = ['#495057', '#6C757D', '#ADB5BD']

    param_colors = na_colors + k_colors + general_colors

    for p in range(num_params):
        ax.plot(data['time_test'], data['ST_time'][:, p],
               label=parameters.get_latex_name(data['param_names'][p]),
               linewidth=2.5, color=param_colors[p], alpha=0.9)

    ax.set_xlabel('Time (ms)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Total-Order Sobol Index ($S_T$)', fontsize=16, fontweight='bold')
    ax.set_title('Time-Dependent Total-Order Sensitivity Indices',
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xlim([0, data['time_test'][-1]])
    ax.set_ylim([0, 1.05])  # Slight headroom above 1
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, frameon=False)
    ax.tick_params(labelsize=13)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Enhance remaining spines
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, config.SOBOL_ST_PNG), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, config.SOBOL_ST_PDF), bbox_inches='tight')
    plt.close()
    print("  ✓ Saved sobol_ST.png and .pdf")


def plot_sobol_ST_at_70ms(data: Dict[str, Any], output_dir: str = '.') -> None:
    """Plot total-order Sobol indices at 12.2 ms and 70 ms as a bar graph."""
    if 'ST_time' not in data or 'time_test' not in data:
        print("  ⊗ Skipping Sobol ST at 70ms plot (no data)")
        return

    print("Plotting Sobol ST at 12.2ms and 70ms...")

    # Find the indices closest to 12.2 ms and 70 ms
    time_array = data['time_test']
    idx_12ms = (np.abs(time_array - 12.2)).argmin()
    idx_70ms = (np.abs(time_array - 70.0)).argmin()
    actual_time_12ms = time_array[idx_12ms]
    actual_time_70ms = time_array[idx_70ms]

    # Extract ST values at those time points
    ST_at_12ms = data['ST_time'][idx_12ms, :]
    ST_at_70ms = data['ST_time'][idx_70ms, :]

    num_params = len(data['param_names'])

    # Get LaTeX parameter names
    param_labels = [parameters.get_latex_name(data['param_names'][p]) for p in range(num_params)]

    fig, ax = plt.subplots(figsize=(18, 7))

    # Set up grouped bar positions
    x_positions = np.arange(num_params)
    bar_width = 0.35

    # Create bars for both time points
    bars1 = ax.bar(x_positions - bar_width/2, ST_at_12ms, bar_width,
                   label=f'{actual_time_12ms:.1f} ms',
                   color='#4363d8', alpha=0.9, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x_positions + bar_width/2, ST_at_70ms, bar_width,
                   label=f'{actual_time_70ms:.1f} ms',
                   color='#e6194b', alpha=0.9, edgecolor='black', linewidth=1.2)

    ax.set_xlabel('Parameter', fontsize=16, fontweight='bold')
    ax.set_ylabel('Total-Order Sobol Index ($S_T$)', fontsize=16, fontweight='bold')
    ax.set_title(f'Total-Order Sensitivity Indices at {actual_time_12ms:.1f} ms and {actual_time_70ms:.1f} ms',
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(param_labels, rotation=45, ha='right', fontsize=11)
    ax.set_ylim([0, 1.0])
    ax.tick_params(axis='y', labelsize=13)
    ax.legend(fontsize=13, frameon=True, loc='upper right')

    # Add grid for easier reading
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Enhance remaining spines
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, config.SOBOL_ST_70MS_PNG), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, config.SOBOL_ST_70MS_PDF), bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved sobol_ST_70ms.png and .pdf (at {actual_time_12ms:.1f} ms and {actual_time_70ms:.1f} ms)")


def plot_sobol_S1_at_70ms(data: Dict[str, Any], output_dir: str = '.') -> None:
    """Plot first-order Sobol indices at 12.2 ms and 70 ms as a bar graph."""
    if 'S1_time' not in data or 'time_test' not in data:
        print("  ⊗ Skipping Sobol S1 at 70ms plot (no data)")
        return

    print("Plotting Sobol S1 at 12.2ms and 70ms...")

    # Find the indices closest to 12.2 ms and 70 ms
    time_array = data['time_test']
    idx_12ms = (np.abs(time_array - 12.2)).argmin()
    idx_70ms = (np.abs(time_array - 70.0)).argmin()
    actual_time_12ms = time_array[idx_12ms]
    actual_time_70ms = time_array[idx_70ms]

    # Extract S1 values at those time points
    S1_at_12ms = data['S1_time'][idx_12ms, :].copy()
    S1_at_70ms = data['S1_time'][idx_70ms, :].copy()

    # Apply offset if first value is negative (same logic as in time series plot)
    num_params = len(data['param_names'])
    for p in range(num_params):
        if data['S1_time'][0, p] < 0:
            offset = data['S1_time'][0, p]
            S1_at_12ms[p] -= offset
            S1_at_70ms[p] -= offset

    # Get LaTeX parameter names
    param_labels = [parameters.get_latex_name(data['param_names'][p]) for p in range(num_params)]

    fig, ax = plt.subplots(figsize=(18, 7))

    # Set up grouped bar positions
    x_positions = np.arange(num_params)
    bar_width = 0.35

    # Create bars for both time points
    bars1 = ax.bar(x_positions - bar_width/2, S1_at_12ms, bar_width,
                   label=f'{actual_time_12ms:.1f} ms',
                   color='#4363d8', alpha=0.9, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x_positions + bar_width/2, S1_at_70ms, bar_width,
                   label=f'{actual_time_70ms:.1f} ms',
                   color='#e6194b', alpha=0.9, edgecolor='black', linewidth=1.2)

    ax.set_xlabel('Parameter', fontsize=16, fontweight='bold')
    ax.set_ylabel('First-Order Sobol Index ($S_1$)', fontsize=16, fontweight='bold')
    ax.set_title(f'First-Order Sensitivity Indices at {actual_time_12ms:.1f} ms and {actual_time_70ms:.1f} ms',
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(param_labels, rotation=45, ha='right', fontsize=11)
    ax.set_ylim([0, 1.0])
    ax.tick_params(axis='y', labelsize=13)
    ax.legend(fontsize=13, frameon=True, loc='upper right')

    # Add grid for easier reading
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Enhance remaining spines
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, config.SOBOL_S1_70MS_PNG), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, config.SOBOL_S1_70MS_PDF), bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved sobol_S1_70ms.png and .pdf (at {actual_time_12ms:.1f} ms and {actual_time_70ms:.1f} ms)")


def plot_combined_A4_layout(data: Dict[str, Any], output_dir: str = '.') -> None:
    """Create A4 landscape layout with all four Sobol plots."""
    if 'S1_time' not in data or 'ST_time' not in data or 'time_test' not in data:
        print("  ⊗ Skipping combined A4 layout (missing data)")
        return

    print("Plotting combined A4 layout...")

    # A4 landscape dimensions in inches (11.69 x 8.27)
    # Increase width by 15% for better spacing: 11.69 * 1.15 ≈ 13.44
    fig = plt.figure(figsize=(13.44, 8.27))

    # Create 2x2 grid of subplots with space for legend on right
    # Adjust right margin to avoid overlap with legend
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3,
                          left=0.05, right=0.75, top=0.9, bottom=0.1)

    # Prepare data
    time_array = data['time_test']
    idx_12ms = (np.abs(time_array - 12.2)).argmin()
    idx_70ms = (np.abs(time_array - 70.0)).argmin()
    actual_time_12ms = time_array[idx_12ms]
    actual_time_70ms = time_array[idx_70ms]

    num_params = len(data['param_names'])
    # Color scheme: hot red for Na (conductance + activation), green for Na inactivation, cool blue for K (conductance + kinetics), gray for others
    param_colors = [
        '#FF0000', '#FF1493', '#8B0000', '#DC143C', '#B22222', '#A0522D', '#800080',  # Na reds/pinks/purples/browns
        '#00FF00', '#32CD32', '#9ACD32', '#FFFF00', '#ADFF2F',  # Na inact greens/yellows
        '#0000FF', '#1E90FF', '#4169E1', '#6495ED', '#87CEEB', '#ADD8E6', '#B0E0E6',  # K lighter blues
        '#808080', '#A0A0A0', '#C0C0C0'  # others grays
    ]

    param_labels = [parameters.get_latex_name(data['param_names'][p]) for p in range(num_params)]

    # --- Top Left: S1 time series ---
    ax1 = fig.add_subplot(gs[0, 0])
    S1_plot = data['S1_time'].copy()
    for p in range(num_params):
        if S1_plot[0, p] < 0:
            S1_plot[:, p] -= S1_plot[0, p]
    for p in range(num_params):
        ax1.plot(time_array, S1_plot[:, p], linewidth=1.5,
                color=param_colors[p], alpha=0.9)
    ax1.set_xlabel('Time [ms]', fontsize=11, fontfamily='Arial')
    ax1.set_ylabel('$S_i$', fontsize=11, fontweight='bold', fontfamily='Arial')
    ax1.set_xlim([0, time_array[-1]])
    ax1.set_ylim(top=1.0)
    ax1.tick_params(labelsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- Top Right: ST time series ---
    ax2 = fig.add_subplot(gs[0, 1])
    for p in range(num_params):
        ax2.plot(time_array, data['ST_time'][:, p], linewidth=1.5,
                color=param_colors[p], alpha=0.9)
    ax2.set_xlabel('Time [ms]', fontsize=11, fontfamily='Arial')
    ax2.set_ylabel('$S_T$', fontsize=11, fontweight='bold', fontfamily='Arial')
    ax2.set_xlim([0, time_array[-1]])
    ax2.set_ylim(top=1.0)
    ax2.tick_params(labelsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Add arrows to top left (S1) and top right (ST) figures
    # Blue bars color: '#4363d8', Red bars color: '#e6194b'
    ax1.arrow(12, 0.4, 0, -0.065, head_width=1.5, head_length=0.026, fc='#4363d8', ec='#4363d8', alpha=0.9)
    ax1.arrow(70, 0.4, 0, -0.065, head_width=1.5, head_length=0.026, fc='#e6194b', ec='#e6194b', alpha=0.9)
    ax2.arrow(12, 0.85, 0, -0.065, head_width=1.5, head_length=0.026, fc='#4363d8', ec='#4363d8', alpha=0.9)
    ax2.arrow(70, 0.85, 0, -0.065, head_width=1.5, head_length=0.026, fc='#e6194b', ec='#e6194b', alpha=0.9)

    # --- Bottom Left: S1 bar graph ---
    ax3 = fig.add_subplot(gs[1, 0])
    S1_at_12ms = data['S1_time'][idx_12ms, :].copy()
    S1_at_70ms = data['S1_time'][idx_70ms, :].copy()
    for p in range(num_params):
        if data['S1_time'][0, p] < 0:
            offset = data['S1_time'][0, p]
            S1_at_12ms[p] -= offset
            S1_at_70ms[p] -= offset
    x_pos = np.arange(num_params)
    bar_width = 0.35
    ax3.bar(x_pos - bar_width/2, S1_at_12ms, bar_width,
           label=f'{actual_time_12ms:.1f} ms', color='#4363d8',
           alpha=0.9, edgecolor='black', linewidth=0.8)
    ax3.bar(x_pos + bar_width/2, S1_at_70ms, bar_width,
           label=f'{actual_time_70ms:.1f} ms', color='#e6194b',
           alpha=0.9, edgecolor='black', linewidth=0.8)
    ax3.set_xlabel('Parameter Name', fontsize=11, fontfamily='Arial')
    ax3.set_ylabel('$S_i$', fontsize=11, fontweight='bold', fontfamily='Arial')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(param_labels, rotation=45, ha='right', fontsize=8, fontweight='bold', fontfamily='Arial')
    ax3.set_ylim([0, 1.0])
    ax3.tick_params(axis='y', labelsize=9)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # --- Bottom Right: ST bar graph ---
    ax4 = fig.add_subplot(gs[1, 1])
    ST_at_12ms = data['ST_time'][idx_12ms, :]
    ST_at_70ms = data['ST_time'][idx_70ms, :]
    ax4.bar(x_pos - bar_width/2, ST_at_12ms, bar_width,
           label=f'{actual_time_12ms:.1f} ms', color='#4363d8',
           alpha=0.9, edgecolor='black', linewidth=0.8)
    ax4.bar(x_pos + bar_width/2, ST_at_70ms, bar_width,
           label=f'{actual_time_70ms:.1f} ms', color='#e6194b',
           alpha=0.9, edgecolor='black', linewidth=0.8)
    ax4.set_xlabel('Parameter Name', fontsize=11, fontfamily='Arial')
    ax4.set_ylabel('$S_T$', fontsize=11, fontweight='bold', fontfamily='Arial')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(param_labels, rotation=45, ha='right', fontsize=8, fontweight='bold', fontfamily='Arial')
    ax4.set_ylim([0, 1.0])
    ax4.tick_params(axis='y', labelsize=9)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # Add parameter legend on the right side
    handles = [plt.Line2D([0], [0], color=param_colors[p], lw=2, label=parameters.get_latex_name(data['param_names'][p])) for p in range(num_params)]
    fig.legend(handles=handles, loc='center right', bbox_to_anchor=(0.85, 0.5), fontsize=8, frameon=False, ncol=1)

    plt.savefig(os.path.join(output_dir, config.SOBOL_COMBINED_A4_PNG), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, config.SOBOL_COMBINED_A4_PDF), bbox_inches='tight')
    plt.close()
    print("  ✓ Saved sobol_combined_A4.png and .pdf")