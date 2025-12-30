"""
Graphics module for the Hodgkin-Huxley cable model.

Contains all plotting functions for visualization of simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from config import DEFAULT_I_AMPLITUDE, INJECTION_DISTANCE, RECORDING_DISTANCE, INJECTION_SEGMENT, RECORDING_SEGMENT
from parameters import get_latex_name


def plot_pie_chart(categories, output_file='trace_categories_pie.png'):
    """
    Create a pie chart showing the relative distribution of trace categories.

    Parameters:
    -----------
    categories : dict
        Categories dictionary from categorize_traces function
    output_file : str
        Output filename for the pie chart
    """
    print("Creating pie chart of trace categories...")

    # Extract data for pie chart
    labels = []
    sizes = []
    colors = ['lightcoral', 'lightblue', 'lightyellow', 'lightgreen', 'orange']

    label_mapping = {
        'no_ap': 'No Action Potential',
        'single_ap_propagated': 'Single AP - Propagated',
        'single_ap_failed': 'Single AP - Failed Propagation',
        'multiple_ap_during_stim': 'Multiple APs During Stimulus',
        'ap_before_stim': 'AP Before Stimulus'
    }

    for category, data in categories.items():
        if data['count'] > 0:
            labels.append(label_mapping[category])
            sizes.append(data['count'])

    if not sizes:
        print("No data to plot in pie chart")
        return

    # Create publication-quality pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    colors_pub = ['#E74C3C', '#3498DB', '#F1C40F', '#2ECC71', '#F39C12']  # Professional colors
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pub[:len(labels)],
                                       autopct='%1.1f%%', startangle=90,
                                       textprops={'fontsize': 13, 'fontweight': 'bold'},
                                       pctdistance=0.85, labeldistance=1.05)

    # Enhance text appearance
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(13)
        autotext.set_fontweight('bold')

    for text in texts:
        text.set_fontsize(13)
        text.set_fontweight('bold')

    ax.set_title('Distribution of Action Potential Patterns',
                fontsize=18, fontweight='bold', pad=20)

    # Add sample size
    total = sum(sizes)
    fig.text(0.5, 0.02, f'Total simulations: n = {total:,}',
            ha='center', fontsize=12, fontweight='bold')

    plt.axis('equal')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(f'trace_categories_pie_I{DEFAULT_I_AMPLITUDE}uA.pdf', bbox_inches='tight')
    plt.close()

    print(f"Pie chart saved to {output_file}")


def plot_representative_traces(categories, time_test, output_file='representative_traces.png'):
    """
    Create a figure with subplots showing representative traces for each category.
    Also saves separate CSV files for each category's representative traces.

    Parameters:
    -----------
    categories : dict
        Categories dictionary from categorize_traces function
    time_test : array
        Time vector
    output_file : str
        Output filename for the figure
    """
    print("Creating representative traces figure...")

    # Filter categories that have traces
    active_categories = {k: v for k, v in categories.items() if v['traces']}

    if not active_categories:
        print("No traces to plot")
        return

    n_categories = len(active_categories)
    fig, axes = plt.subplots(n_categories, 1, figsize=(12, 3.5 * n_categories))

    if n_categories == 1:
        axes = [axes]

    label_mapping = {
        'no_ap': 'No Action Potential',
        'single_ap_propagated': 'Single AP - Propagated',
        'single_ap_failed': 'Single AP - Failed Propagation',
        'multiple_ap_during_stim': 'Sustained Repetitive Firing',
        'ap_before_stim': 'AP Before Stimulus'
    }

    colors_pub = ['#E74C3C', '#3498DB', '#F1C40F', '#2ECC71', '#F39C12', '#9B59B6']

    for idx, (category, data) in enumerate(active_categories.items()):
        ax = axes[idx]

        # Plot all representative traces for this category
        for i, trace in enumerate(data['traces']):
            alpha = 0.8 if i == 0 else 0.6
            linewidth = 2 if i == 0 else 1.5
            ax.plot(time_test, trace, color=colors_pub[i % len(colors_pub)],
                   alpha=alpha, linewidth=linewidth,
                   label=f'Example {i+1}' if i < 3 else None)

        # Highlight stimulus period
        ax.axvspan(10.0, 80.0, alpha=0.15, color='gold', label='Stimulus' if idx == 0 else None, zorder=0)

        # Formatting for publication
        ax.set_ylabel('Membrane Potential (mV)', fontsize=12, fontweight='bold')
        ax.set_title(f'{label_mapping[category]} (n={data["count"]:,})',
                    fontsize=13, fontweight='bold', pad=10)
        ax.set_xlim([0, time_test[-1]])
        ax.tick_params(labelsize=11)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['left'].set_linewidth(1.2)

        # Add legend only for first subplot
        if idx == 0 and len(data['traces']) > 1:
            ax.legend(fontsize=10, loc='upper right', framealpha=0.9)

    # Set x-label only for bottom subplot
    axes[-1].set_xlabel('Time (ms)', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(f'representative_traces_I{DEFAULT_I_AMPLITUDE}uA.pdf', bbox_inches='tight')
    plt.close()

    print(f"Representative traces figure saved to {output_file}")

    # Save CSV files for each category
    print("Saving CSV files for representative traces...")
    for category, data in active_categories.items():
        if data['traces']:
            # Create array with time column and trace columns
            # Shape: (time_points, 1 + num_traces)
            csv_data = np.column_stack([time_test] + data['traces'])

            # Generate header
            header = 'time_ms,' + ','.join([f'trace_{i+1}' for i in range(len(data['traces']))])

            # Save to CSV
            csv_filename = f'representative_{category}.csv'
            np.savetxt(csv_filename, csv_data, delimiter=',', header=header, comments='')
            print(f"  Saved {csv_filename} ({len(data['traces'])} traces)")

    print("Representative trace CSV files saved successfully")


def plot_parameter_scatter_matrix(categories, param_values, param_names, param_bounds):
    """
    Create a 22x22 scatter plot matrix for parameters in the 'single_ap_propagated' category.
    Each subplot shows parameter i vs parameter j.

    Parameters:
    -----------
    categories : dict
        Categories dictionary from categorize_traces function
    param_values : array
        All parameter values used in simulations (N × num_params)
    param_names : list
        Names of parameters
    param_bounds : list of tuples
        (min, max) bounds for each parameter
    """
    print("\nCreating 22x22 parameter scatter matrix for single_ap_propagated category...")

    # Get data for single_ap_propagated category
    if 'single_ap_propagated' not in categories or categories['single_ap_propagated']['count'] == 0:
        print("No traces in single_ap_propagated category - skipping scatter matrix")
        return

    category_data = categories['single_ap_propagated']
    category_params = param_values[category_data['param_indices'], :]
    n_samples = category_params.shape[0]
    n_params = len(param_names)

    print(f"  Category: single_ap_propagated")
    print(f"  Number of samples: {n_samples}")
    print(f"  Number of parameters: {n_params}")

    # Subsample if we have too many points for clear visualization
    max_plot_points = 2000
    if n_samples > max_plot_points:
        print(f"  Subsampling {max_plot_points} points from {n_samples} for clearer visualization...")
        np.random.seed(42)
        sample_indices = np.random.choice(n_samples, max_plot_points, replace=False)
        plot_params = category_params[sample_indices, :]
        n_plot = max_plot_points
    else:
        plot_params = category_params
        n_plot = n_samples

    print(f"  Plotting {n_plot} points")

    # Create figure with 22x22 subplots
    fig, axes = plt.subplots(n_params, n_params, figsize=(44, 44))

    print("  Generating scatter plots...")

    # Create scatter plots
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]

            if i == j:
                # Diagonal: histogram of parameter i (use full dataset, not subsampled)
                ax.hist(category_params[:, i], bins=30, alpha=0.7, color='steelblue',
                       edgecolor='black', linewidth=0.3, range=param_bounds[i])
                ax.set_xlim(param_bounds[i])
            else:
                # Off-diagonal: scatter plot of parameter j vs parameter i
                # Use larger marker size and lower alpha for better visibility
                ax.scatter(plot_params[:, j], plot_params[:, i],
                          s=3, alpha=0.5, color='steelblue', edgecolors='none', rasterized=True)
                ax.set_xlim(param_bounds[j])
                ax.set_ylim(param_bounds[i])

            # Labels only on edges
            if j == 0:  # Left edge
                ax.set_ylabel(get_latex_name(param_names[i]), fontsize=8, fontweight='bold')
            else:
                ax.set_yticklabels([])

            if i == n_params - 1:  # Bottom edge
                ax.set_xlabel(get_latex_name(param_names[j]), fontsize=8, fontweight='bold')
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                ax.set_xticklabels([])

            # Smaller tick labels
            ax.tick_params(labelsize=6)

        # Progress update
        if (i + 1) % 5 == 0:
            print(f"    Progress: {i+1}/{n_params} rows complete")

    # Overall title
    if n_plot < n_samples:
        title_text = f'Parameter Scatter Matrix: Single AP Propagated\n(Total n={n_samples:,}, plotted {n_plot:,} points)'
    else:
        title_text = f'Parameter Scatter Matrix: Single AP Propagated (n={n_samples:,})'
    fig.suptitle(title_text, fontsize=24, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save figure
    print("  Saving figure (this may take a moment due to large size)...")
    output_file = 'parameter_scatter_matrix_propagated.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')  # Lower DPI for manageable file size
    print(f"  ✓ Saved {output_file}")

    # Also save PDF
    output_pdf = 'parameter_scatter_matrix_propagated.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"  ✓ Saved {output_pdf}")

    plt.close()
    print("  Scatter matrix plot complete")


def plot_parameter_distributions_by_category(categories, param_values, param_names, param_bounds):
    """
    Plot parameter distributions for each firing category.
    Creates a multi-panel figure with parameter histograms for each category.

    Parameters:
    -----------
    categories : dict
        Categories dictionary from categorize_traces function
    param_values : array
        All parameter values used in simulations (N × num_params)
    param_names : list
        Names of parameters
    param_bounds : list of tuples
        (min, max) bounds for each parameter
    """
    print("Creating parameter distribution plots for each category...")

    label_mapping = {
        'no_ap': 'No Action Potential',
        'single_ap_propagated': 'Single AP - Propagated',
        'single_ap_failed': 'Single AP - Failed Propagation',
        'multiple_ap_during_stim': 'Multiple APs',
        'ap_before_stim': 'AP Before Stimulus'
    }

    # Filter out categories with no traces
    active_categories = {k: v for k, v in categories.items() if v['count'] > 0}

    # Create plots for each category
    for category, data in active_categories.items():
        if len(data['param_indices']) == 0:
            continue

        # Get parameters for this category
        category_params = param_values[data['param_indices'], :]
        n_params = len(param_names)

        # Pre-calculate all histograms to find max y-value for this category
        all_counts = []
        for i in range(n_params):
            param_data = category_params[:, i]
            bounds = param_bounds[i]
            counts, _ = np.histogram(param_data, bins=30, range=bounds)
            all_counts.append(counts)

        # Find max count for y-axis scaling within this category
        max_count = max(np.max(c) for c in all_counts)
        y_max = max_count * 1.1  # Add 10% headroom

        print(f"  Category '{category}': n={data['count']}, max_count={max_count}, y_max={y_max:.1f}")

        # Create figure with subplots for each parameter
        fig, axes = plt.subplots(6, 4, figsize=(16, 18))
        axes = axes.flatten()

        for i, (param_name, bounds) in enumerate(zip(param_names, param_bounds)):
            ax = axes[i]
            param_data = category_params[:, i]

            # Create histogram with full parameter range
            ax.hist(param_data, bins=30, alpha=0.7, color='steelblue',
                   edgecolor='black', linewidth=0.5, range=bounds)

            # Formatting for publication
            ax.set_xlabel(get_latex_name(param_name), fontsize=10, fontweight='bold')
            ax.set_ylabel('Count', fontsize=10)
            ax.set_xlim(bounds)
            ax.set_ylim(0, y_max)  # Use category-specific y-scale
            ax.tick_params(labelsize=9)

            # Add mean line
            mean_val = np.mean(param_data)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        # Hide extra subplots
        for i in range(n_params, len(axes)):
            axes[i].axis('off')

        # Overall title
        title = f'{label_mapping[category]} (n={data["count"]:,})'
        fig.suptitle(f'Parameter Distributions: {title}',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout(rect=[0, 0, 1, 0.99])

        # Save figure
        output_file = f'parameter_distributions_{category}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(f'parameter_distributions_{category}_I{DEFAULT_I_AMPLITUDE}uA.pdf', bbox_inches='tight')
        plt.close()

        print(f"  Saved {output_file} and PDF version")

    print("Parameter distribution plots complete")


def plot_firing_frequency_histogram(firing_frequencies, output_file='firing_frequency_histogram.png'):
    """
    Create a histogram of firing frequencies for traces with multiple action potentials.

    Parameters:
    -----------
    firing_frequencies : array
        Array of firing frequencies (Hz)
    output_file : str
        Output filename for the histogram
    """
    print("Creating firing frequency histogram...")

    if len(firing_frequencies) == 0:
        print("No firing frequency data to plot")
        return

    # Create publication-quality histogram
    fig, ax = plt.subplots(figsize=(9, 6))
    counts, bins, patches = ax.hist(firing_frequencies, bins=100, alpha=0.75,
                                     edgecolor='black', linewidth=0.8, color='steelblue')
    ax.set_xlabel('Firing Frequency (Hz)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Simulations', fontsize=14, fontweight='bold')
    ax.set_title('Distribution of Firing Frequencies\n(Sustained Repetitive Firing Throughout Stimulus)',
                fontsize=16, fontweight='bold', pad=15)
    ax.tick_params(labelsize=12)

    # Add statistics
    mean_freq = np.mean(firing_frequencies)
    median_freq = np.median(firing_frequencies)
    std_freq = np.std(firing_frequencies)

    # Statistics box
    stats_text = f'Mean: {mean_freq:.1f} Hz\nMedian: {median_freq:.1f} Hz\nSD: {std_freq:.1f} Hz\nn = {len(firing_frequencies):,}'
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black'))

    # Set x-axis limit based on data (95th percentile with some margin)
    p95 = np.percentile(firing_frequencies, 95)
    max_freq = np.max(firing_frequencies)
    # Use 95th percentile + 10% margin, but cap at reasonable max
    xlim_max = min(p95 * 1.1, max_freq * 1.05)
    ax.set_xlim(0, xlim_max)

    # Add mean and median lines
    ax.axvline(mean_freq, color='red', linestyle='--', linewidth=2,
                label='Mean', alpha=0.8)
    ax.axvline(median_freq, color='orange', linestyle='-.', linewidth=2,
                label='Median', alpha=0.8)

    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)

    # Spine formatting
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(f'firing_frequency_histogram_I{DEFAULT_I_AMPLITUDE}uA.pdf', bbox_inches='tight')
    plt.close()

    print(f"Firing frequency histogram saved to {output_file}")


def plot_stable_potential_histogram(stable_potentials, measurement_time=80.0, output_file='stable_potential_histogram.png'):
    """
    Create a histogram of stable membrane potentials at a specific time point.

    Parameters:
    -----------
    stable_potentials : array
        Array of membrane potentials (mV) at measurement_time
    measurement_time : float
        Time point when potentials were measured (ms)
    output_file : str
        Output filename for the histogram
    """
    print("Creating stable potential histogram...")

    if len(stable_potentials) == 0:
        print("No stable potential data to plot")
        return

    # Create publication-quality histogram
    fig, ax = plt.subplots(figsize=(9, 6))
    counts, bins, patches = ax.hist(stable_potentials, bins=60, alpha=0.75,
                                     edgecolor='black', linewidth=0.8, color='coral')
    ax.set_xlabel('Membrane Potential (mV)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Simulations', fontsize=14, fontweight='bold')
    ax.set_title(f'Membrane Potential at t = {measurement_time} ms\n(Single AP After Stimulus)',
                fontsize=16, fontweight='bold', pad=15)
    ax.tick_params(labelsize=12)

    # Add statistics
    mean_potential = np.mean(stable_potentials)
    median_potential = np.median(stable_potentials)
    std_potential = np.std(stable_potentials)

    # Statistics box
    stats_text = f'Mean: {mean_potential:.2f} mV\nMedian: {median_potential:.2f} mV\nSD: {std_potential:.2f} mV\nn = {len(stable_potentials):,}'
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black'))

    # Add mean and median lines
    ax.axvline(mean_potential, color='red', linestyle='--', linewidth=2,
                label='Mean', alpha=0.8)
    ax.axvline(median_potential, color='orange', linestyle='-.', linewidth=2,
                label='Median', alpha=0.8)

    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)

    # Spine formatting
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(f'stable_potential_histogram_I{DEFAULT_I_AMPLITUDE}uA.pdf', bbox_inches='tight')
    plt.close()

    print(f"Stable potential histogram saved to {output_file}")


def plot_ap_amplitude_histograms(propagation_results, output_file_prefix='ap_amplitude'):
    """
    Create histograms of AP amplitudes at injection and distal sites.

    Parameters:
    -----------
    propagation_results : dict
        Results from analyze_ap_propagation function
    output_file_prefix : str
        Prefix for output filenames
    """
    amplitudes_proximal = propagation_results['amplitudes_proximal']
    amplitudes_distal = propagation_results['amplitudes_distal']

    # --- HISTOGRAM 1: AP Amplitude at Injection Site ---
    if len(amplitudes_proximal) > 0:
        print("Creating AP amplitude histogram at injection site (X=0.4 cm)...")
        fig, ax = plt.subplots(figsize=(9, 6))
        counts, bins, patches = ax.hist(amplitudes_proximal, bins=50, alpha=0.75,
                                         edgecolor='black', linewidth=0.8, color='steelblue')
        ax.set_xlabel('AP Amplitude (mV)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Simulations', fontsize=14, fontweight='bold')
        ax.set_title(f'AP Amplitude at Injection Site (X={INJECTION_DISTANCE} cm)\n(Single AP After Stimulus)',
                    fontsize=16, fontweight='bold', pad=15)
        ax.tick_params(labelsize=12)

        # Statistics
        mean_amp = np.mean(amplitudes_proximal)
        median_amp = np.median(amplitudes_proximal)
        std_amp = np.std(amplitudes_proximal)

        stats_text = f'Mean: {mean_amp:.1f} mV\nMedian: {median_amp:.1f} mV\nSD: {std_amp:.1f} mV\nn = {len(amplitudes_proximal):,}'
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black'))

        ax.axvline(mean_amp, color='red', linestyle='--', linewidth=2, label='Mean', alpha=0.8)
        ax.axvline(median_amp, color='orange', linestyle='-.', linewidth=2, label='Median', alpha=0.8)
        ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['left'].set_linewidth(1.2)

        plt.tight_layout()
        plt.savefig(f'{output_file_prefix}_proximal_histogram.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_file_prefix}_proximal_histogram_I{DEFAULT_I_AMPLITUDE}uA.pdf', bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved {output_file_prefix}_proximal_histogram.png/pdf")

    # --- HISTOGRAM 2: AP Amplitude at Distal Site ---
    if len(amplitudes_distal) > 0:
        print("\nCreating AP amplitude histogram at distal site (X=8.0 cm)...")
        fig, ax = plt.subplots(figsize=(9, 6))
        counts, bins, patches = ax.hist(amplitudes_distal, bins=50, alpha=0.75,
                                         edgecolor='black', linewidth=0.8, color='coral')
        ax.set_xlabel('AP Amplitude (mV)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Simulations', fontsize=14, fontweight='bold')
        ax.set_title(f'AP Amplitude at Distal Site (X={RECORDING_DISTANCE} cm)\n(Single AP After Stimulus)',
                    fontsize=16, fontweight='bold', pad=15)
        ax.tick_params(labelsize=12)

        # Statistics
        mean_amp = np.mean(amplitudes_distal)
        median_amp = np.median(amplitudes_distal)
        std_amp = np.std(amplitudes_distal)

        stats_text = f'Mean: {mean_amp:.1f} mV\nMedian: {median_amp:.1f} mV\nSD: {std_amp:.1f} mV\nn = {len(amplitudes_distal):,}'
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black'))

        # Add failure threshold line
        failure_threshold = 10.0  # From config
        ax.axvline(failure_threshold, color='black', linestyle=':', linewidth=2.5,
                  label=f'Failure Threshold ({failure_threshold} mV)', alpha=0.7)
        ax.axvline(mean_amp, color='red', linestyle='--', linewidth=2, label='Mean', alpha=0.8)
        ax.axvline(median_amp, color='orange', linestyle='-.', linewidth=2, label='Median', alpha=0.8)
        ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['left'].set_linewidth(1.2)

        plt.tight_layout()
        plt.savefig(f'{output_file_prefix}_distal_histogram.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_file_prefix}_distal_histogram_I{DEFAULT_I_AMPLITUDE}uA.pdf', bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved {output_file_prefix}_distal_histogram.png/pdf")


def plot_propagation_speed_histogram(propagation_speeds, output_file='propagation_speed_histogram.png'):
    """
    Create a histogram of propagation speeds for successfully propagated APs.

    Parameters:
    -----------
    propagation_speeds : array
        Array of propagation speeds (m/s)
    output_file : str
        Output filename for the histogram
    """
    print("Creating propagation speed histogram...")

    if len(propagation_speeds) == 0:
        print("No propagation speed data to plot")
        return

    # Filter out outliers above 20 m/s for histogram
    speeds_filtered = propagation_speeds[propagation_speeds < 20]
    n_outliers = len(propagation_speeds) - len(speeds_filtered)

    if n_outliers > 0:
        print(f"  Note: Filtered {n_outliers} outliers (speed >= 20 m/s) from histogram")

    fig, ax = plt.subplots(figsize=(9, 6))
    counts, bins, patches = ax.hist(speeds_filtered, bins=50, alpha=0.75,
                                     edgecolor='black', linewidth=0.8, color='mediumseagreen')
    ax.set_xlabel('Propagation Speed (m/s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Simulations', fontsize=14, fontweight='bold')
    ax.set_title(f'Action Potential Propagation Speed\n(Successful Propagation Only)',
                fontsize=16, fontweight='bold', pad=15)
    ax.set_xlim(0, 20)  # Set x-axis range to 0-20 m/s
    ax.tick_params(labelsize=12)

    # Statistics (calculated on filtered data for display)
    mean_speed = np.mean(speeds_filtered)
    median_speed = np.median(speeds_filtered)
    std_speed = np.std(speeds_filtered)

    stats_text = f'Mean: {mean_speed:.2f} m/s\nMedian: {median_speed:.2f} m/s\nSD: {std_speed:.2f} m/s\nn = {len(speeds_filtered):,}'
    if n_outliers > 0:
        stats_text += f'\n({n_outliers} outliers excluded)'
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black'))

    ax.axvline(mean_speed, color='red', linestyle='--', linewidth=2, label='Mean', alpha=0.8)
    ax.axvline(median_speed, color='orange', linestyle='-.', linewidth=2, label='Median', alpha=0.8)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(f'propagation_speed_histogram_I{DEFAULT_I_AMPLITUDE}uA.pdf', bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved {output_file}")


def plot_single_ap_traces_overlay(time_test, traces_inj, traces_dist, max_traces=500,
                                  output_file_prefix='single_ap_traces'):
    """
    Create overlay plots of representative single AP traces at injection and distal sites.

    Parameters:
    -----------
    time_test : array
        Time vector
    traces_inj : array
        Single AP traces at injection site
    traces_dist : array
        Single AP traces at distal site
    max_traces : int
        Maximum number of traces to plot
    output_file_prefix : str
        Prefix for output filenames
    """
    if len(traces_inj) == 0:
        print("No single AP traces to plot")
        return

    # Filter traces: only keep those where V(70ms) < -45 mV
    # Find the index closest to 70 ms
    time_70ms_idx = np.argmin(np.abs(time_test - 70.0))

    filtered_traces_inj = []
    filtered_traces_dist = []
    for i, (trace_inj, trace_dist) in enumerate(zip(traces_inj, traces_dist)):
        if trace_inj[time_70ms_idx] < -45.0:
            filtered_traces_inj.append(trace_inj)
            filtered_traces_dist.append(trace_dist)

    n_filtered = len(filtered_traces_inj)
    print(f"  Filtered {n_filtered} traces (V(70ms) < -45 mV) out of {len(traces_inj)} total")

    # Limit to max_traces representative traces
    if n_filtered > max_traces:
        # Randomly sample max_traces traces
        import random
        indices = random.sample(range(n_filtered), max_traces)
        traces_inj_to_plot = [filtered_traces_inj[i] for i in indices]
        traces_dist_to_plot = [filtered_traces_dist[i] for i in indices]
        n_plotted = max_traces
    else:
        traces_inj_to_plot = filtered_traces_inj
        traces_dist_to_plot = filtered_traces_dist
        n_plotted = n_filtered

    # Plot 1: Representative traces at injection site (X=0.4 cm)
    # Aspect ratio 1:1 means equal scaling: 100 ms (x) = 140 mV (y)
    fig, ax = plt.subplots(figsize=(10, 14))
    for trace in traces_inj_to_plot:
        ax.plot(time_test, trace, alpha=0.3, linewidth=0.5, color='steelblue')

    ax.axvline(10.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Stimulus Start')
    ax.axvline(80.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Stimulus End')
    ax.set_xlabel('Time (ms)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Membrane Potential (mV)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(-80, 60)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'Single AP Traces at Injection Site (X={INJECTION_DISTANCE} cm)\n{n_plotted} of {n_filtered} filtered simulations (V(70ms) < -45 mV)',
                fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{output_file_prefix}_injection_site.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_file_prefix}_injection_site_I{DEFAULT_I_AMPLITUDE}uA.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {output_file_prefix}_injection_site.png/pdf")

    # Plot 2: Representative traces at distal site (X=8.0 cm)
    fig, ax = plt.subplots(figsize=(10, 14))
    for trace in traces_dist_to_plot:
        ax.plot(time_test, trace, alpha=0.3, linewidth=0.5, color='forestgreen')

    ax.axvline(10.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Stimulus Start')
    ax.axvline(80.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Stimulus End')
    ax.set_xlabel('Time (ms)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Membrane Potential (mV)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(-80, 60)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'Single AP Traces at Distal Site (X={RECORDING_DISTANCE} cm)\n{n_plotted} of {n_filtered} filtered simulations (V(70ms) < -45 mV)',
                fontsize=16, fontweight='bold', pad=15)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{output_file_prefix}_distal_site.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_file_prefix}_distal_site_I{DEFAULT_I_AMPLITUDE}uA.pdf', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {output_file_prefix}_distal_site.png/pdf")


def plot_mean_trace_with_std(time_test, mean_V, std_V, output_file='mean_trace_with_std.png'):
    """
    Plot the mean voltage trace with standard deviation shading.

    Parameters:
    -----------
    time_test : array
        Time vector
    mean_V : array
        Mean voltage trace
    std_V : array
        Standard deviation of voltage traces
    output_file : str
        Output filename for the plot
    """
    print("Creating mean trace with standard deviation plot...")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot mean trace
    ax.plot(time_test, mean_V, color='blue', linewidth=2, label='Mean')

    # Add standard deviation shading
    ax.fill_between(time_test, mean_V - std_V, mean_V + std_V,
                    alpha=0.3, color='blue', label='±1 SD')

    # Highlight stimulus period
    ax.axvspan(10.0, 80.0, alpha=0.15, color='gold', label='Stimulus')

    # Formatting
    ax.set_xlabel('Time (ms)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Membrane Potential (mV)', fontsize=14, fontweight='bold')
    ax.set_title('Mean Voltage Trace with Standard Deviation',
                fontsize=16, fontweight='bold', pad=15)
    ax.set_xlim([0, time_test[-1]])
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9)

    # Spine formatting
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(f'mean_trace_with_std_I{DEFAULT_I_AMPLITUDE}uA.pdf', bbox_inches='tight')
    plt.close()

    print(f"Mean trace plot saved to {output_file}")