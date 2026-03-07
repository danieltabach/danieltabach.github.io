"""
Staggered Difference-in-Differences Event-Study Simulation
A comprehensive tutorial on handling staggered treatment adoption in causal inference.

This script generates synthetic panel data for a multi-location service organization
rolling out new operational software, with staggered adoption times across locations.
It produces publication-quality visualizations and event-study regression results.

Author: Danny Tabach
Date: 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# 1. GENERATE SYNTHETIC PANEL DATA
# ============================================================================

def generate_synthetic_data(n_locations=200, n_treatment=120, n_control=80, n_months=24):
    """
    Generate synthetic panel data for staggered adoption event study.

    Parameters:
    -----------
    n_locations : int
        Total number of locations
    n_treatment : int
        Number of treatment locations
    n_control : int
        Number of control locations
    n_months : int
        Total months in panel

    Returns:
    --------
    data : pd.DataFrame
        Panel data with columns: location_id, month, employee_id,
        metric, treatment, adoption_start, training_start, training_end
    """

    records = []

    # Treatment locations
    for loc_id in range(1, n_treatment + 1):
        # Random adoption start month (between months 3-15)
        adoption_start = np.random.randint(3, 16)

        # Random number of employees (1-5)
        n_employees = np.random.randint(1, 6)

        # Employee training start times (staggered within location)
        # Some start at the same time, some spread out over months
        if np.random.random() < 0.4:
            # 40% of locations train everyone at once
            employee_starts = [adoption_start] * n_employees
        else:
            # 60% stagger training over 1-3 months
            stagger_months = np.random.randint(1, 4)
            employee_starts = [adoption_start + np.random.randint(0, stagger_months)
                              for _ in range(n_employees)]

        # Each employee takes 6 months to complete training
        employee_completions = [start + 6 for start in employee_starts]

        # Adoption window = first completion to last completion
        first_completion = min(employee_completions)
        last_completion = max(employee_completions)
        adoption_window = last_completion - first_completion

        # Filter: keep only if adoption window <= 6 months
        if adoption_window > 6:
            continue

        # Generate monthly observations for this location
        for month in range(1, n_months + 1):
            # Baseline metric with slight upward trend and noise
            baseline = 50
            trend = 0.3 * month
            noise = np.random.normal(0, 2)

            # Post-treatment effect
            post_effect = 0

            # Determine if location is in post-adoption period
            if month >= first_completion:
                # Effect ramps up: +1 in first month, +2 in second, +3+
                months_since = min(month - first_completion + 1, 3)
                post_effect = months_since * 1.5  # Ramp-up effect

            metric = baseline + trend + post_effect + noise

            records.append({
                'location_id': loc_id,
                'month': month,
                'metric': metric,
                'treatment': 1,
                'adoption_start': adoption_start,
                'first_completion': first_completion,
                'last_completion': last_completion,
                'adoption_window': adoption_window,
                'n_employees': n_employees
            })

    # Control locations (don't adopt during observation window)
    for loc_id in range(n_treatment + 1, n_locations + 1):
        for month in range(1, n_months + 1):
            baseline = 50
            trend = 0.3 * month
            noise = np.random.normal(0, 2)
            metric = baseline + trend + noise

            records.append({
                'location_id': loc_id,
                'month': month,
                'metric': metric,
                'treatment': 0,
                'adoption_start': np.nan,
                'first_completion': np.nan,
                'last_completion': np.nan,
                'adoption_window': np.nan,
                'n_employees': np.nan
            })

    data = pd.DataFrame(records)

    # Report filtering results
    n_treatment_qualified = data[data['treatment'] == 1]['location_id'].nunique()
    print(f"Treatment locations with adoption_window <= 6 months: {n_treatment_qualified}")
    print(f"Total locations in final dataset: {data['location_id'].nunique()}")

    return data

# ============================================================================
# 2. EVENT-TIME NORMALIZATION
# ============================================================================

def normalize_to_event_time(data):
    """
    Normalize calendar time to relative event time (periods relative to adoption).

    For treatment: Period 0 = adoption window (first completion)
    For control: assign pseudo-event times based on median treatment timing

    Parameters:
    -----------
    data : pd.DataFrame
        Panel data

    Returns:
    --------
    data : pd.DataFrame
        Data with additional columns: event_time, period
    """

    # Get median treatment adoption time (first completion month)
    median_adoption = data[data['treatment'] == 1]['first_completion'].median()

    data['event_time'] = np.nan
    data['period'] = np.nan

    # Treatment locations: event_time = calendar_month - first_completion month
    treatment_data = data[data['treatment'] == 1].copy()
    for idx, row in treatment_data.iterrows():
        event_time = row['month'] - row['first_completion']
        data.loc[idx, 'event_time'] = event_time

        # Assign period: -6 to -1 (pre), 0 (adoption window), 1 to 6+ (post)
        if -6 <= event_time <= 6:
            data.loc[idx, 'period'] = int(event_time)

    # Control locations: use pseudo-event time (calendar month - median treatment adoption)
    control_mask = data['treatment'] == 0
    data.loc[control_mask, 'event_time'] = data.loc[control_mask, 'month'] - median_adoption

    # Same period binning for control
    control_data = data[control_mask].copy()
    for idx, row in control_data.iterrows():
        event_time = row['event_time']
        if -6 <= event_time <= 6:
            data.loc[idx, 'period'] = int(event_time)

    return data

# ============================================================================
# 3. REGRESSION AND COEFFICIENT EXTRACTION
# ============================================================================

def run_event_study_regression(data):
    """
    Run event-study regression with location and month fixed effects.

    Y_it = alpha_i + gamma_t + sum(beta_k * D_it^k) + epsilon_it

    Parameters:
    -----------
    data : pd.DataFrame
        Panel data with event_time, period, and metric columns

    Returns:
    --------
    results : regression results object
    coefficients : dict of {period: coefficient}
    ci_lower : dict of {period: ci_lower}
    ci_upper : dict of {period: ci_upper}
    """

    # Filter to periods -6 to +6 with non-null data
    reg_data = data[(data['period'].notna()) &
                    (data['period'] >= -6) &
                    (data['period'] <= 6)].copy()

    # Create treatment x period interaction dummies (D_it^k in the model)
    # Each dummy = 1 only if location is treatment AND in period k
    for period in range(-6, 7):
        reg_data[f'period_{period}'] = (
            (reg_data['period'] == period) & (reg_data['treatment'] == 1)
        ).astype(int)

    # Drop period -1 (reference period for omitted variable)
    reg_data = reg_data.drop(columns=['period_-1'])

    # Create fixed effects indicators
    location_dummies = pd.get_dummies(reg_data['location_id'], prefix='loc', drop_first=True)
    month_dummies = pd.get_dummies(reg_data['month'], prefix='month', drop_first=True)

    # Prepare regression data
    Y = reg_data['metric'].astype(float)
    X = reg_data[[col for col in reg_data.columns if col.startswith('period_')]].astype(float)
    X = pd.concat([X, location_dummies.astype(float), month_dummies.astype(float)], axis=1)

    # Add constant
    X = sm.add_constant(X, has_constant='add')

    # Run regression
    model = sm.OLS(Y, X)
    results = model.fit()

    print("\n" + "="*70)
    print("EVENT-STUDY REGRESSION RESULTS")
    print("="*70)
    print(results.summary())

    # Extract coefficients for event periods
    coefficients = {}
    ci_lower = {}
    ci_upper = {}

    periods = list(range(-6, 0)) + list(range(1, 7))

    for period in periods:
        col_name = f'period_{period}'
        if col_name in results.params.index:
            coef = results.params[col_name]
            ci = results.conf_int().loc[col_name]
            coefficients[period] = coef
            ci_lower[period] = ci[0]
            ci_upper[period] = ci[1]
        else:
            # Period -1 is reference (omitted)
            coefficients[period] = 0
            ci_lower[period] = 0
            ci_upper[period] = 0

    return results, coefficients, ci_lower, ci_upper

# ============================================================================
# 4. PLOTTING FUNCTIONS
# ============================================================================

def setup_plot_style():
    """Configure matplotlib style for clean, professional plots."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # Custom rcParams
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = '#f8f8f8'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linewidth'] = 0.5

def plot_staggered_adoption_heatmap(data, output_path, n_locs_show=18):
    """
    Plot heatmap showing staggered adoption across locations and time.

    Color-code: white=no activity, yellow=training, green=completed, blue=post-adoption
    """
    setup_plot_style()

    # Select subset of treatment locations for visualization
    treatment_locs = data[data['treatment'] == 1]['location_id'].unique()[:n_locs_show]

    heatmap_data = []
    location_labels = []

    for loc_id in treatment_locs:
        loc_data = data[data['location_id'] == loc_id].copy()
        loc_data = loc_data.sort_values('month')

        first_comp = loc_data['first_completion'].iloc[0]

        row = []
        for month in range(1, 25):
            month_data = loc_data[loc_data['month'] == month]
            if len(month_data) == 0:
                row.append(0)
                continue

            if month < first_comp:
                row.append(0)  # white - no activity
            elif month == first_comp:
                row.append(1)  # yellow - completion month
            elif month < first_comp + 3:
                row.append(2)  # green - recently completed
            else:
                row.append(3)  # blue - post-adoption

        heatmap_data.append(row)
        location_labels.append(f"Loc {loc_id}")

    heatmap_data = np.array(heatmap_data)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Create custom colormap
    colors = ['white', '#FFE66D', '#95E1D3', '#4A90E2']
    n_bins = len(colors)
    cmap = plt.cm.colors.ListedColormap(colors)

    im = ax.imshow(heatmap_data, aspect='auto', cmap=cmap, interpolation='nearest')

    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f'M{i+1}' for i in range(0, 24, 2)])
    ax.set_yticks(range(len(location_labels)))
    ax.set_yticklabels(location_labels, fontsize=9)

    ax.set_xlabel('Calendar Month', fontsize=11, fontweight='bold')
    ax.set_ylabel('Location', fontsize=11, fontweight='bold')
    ax.set_title('The Reality: Staggered Adoption Across Locations',
                 fontsize=13, fontweight='bold', pad=15)

    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='white', markersize=10, label='No Activity'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#FFE66D', markersize=10, label='Training'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#95E1D3', markersize=10, label='Completed'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#4A90E2', markersize=10, label='Post-Adoption')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_naive_did_problem(data, output_path):
    """
    Two-panel figure showing why naive DiD fails with staggered adoption.
    Left: Naive approach (single cutoff) appears to show effect
    Right: Stacked area chart showing treatment group composition over time
    """
    setup_plot_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # LEFT PANEL: Naive DiD view (single cutoff at month 12)
    cutoff_month = 12

    # Aggregate by treatment status and time period
    data_agg = data.groupby(['month', 'treatment'])['metric'].agg(['mean', 'std', 'count']).reset_index()

    for treatment in [0, 1]:
        subset = data_agg[data_agg['treatment'] == treatment]
        color = '#4A90E2' if treatment == 1 else '#999999'
        label = 'Treatment' if treatment == 1 else 'Control'
        ax1.plot(subset['month'], subset['mean'], marker='o', linewidth=2,
                label=label, color=color, markersize=5)

    ax1.axvline(x=cutoff_month, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Naive Cutoff')
    ax1.fill_between([cutoff_month, 24], 48, 56, alpha=0.1, color='red', label='Post Period')
    ax1.set_xlabel('Calendar Month', fontweight='bold')
    ax1.set_ylabel('Average Metric', fontweight='bold')
    ax1.set_title('Naive DiD: Single Cutoff Approach', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([48, 56])

    # RIGHT PANEL: Stacked area chart of treatment group composition
    treatment_data = data[data['treatment'] == 1].copy()
    treatment_locs = treatment_data.drop_duplicates('location_id')

    months = range(1, 25)
    pre_counts = []
    training_counts = []
    post_counts = []

    for month in months:
        n_pre = 0
        n_training = 0
        n_post = 0
        for _, loc in treatment_locs.iterrows():
            fc = loc['first_completion']
            lc = loc['last_completion']
            a_start = loc['adoption_start']
            if month < a_start:
                n_pre += 1
            elif month >= lc:
                n_post += 1
            else:
                n_training += 1
        pre_counts.append(n_pre)
        training_counts.append(n_training)
        post_counts.append(n_post)

    months_list = list(months)
    ax2.fill_between(months_list, 0, pre_counts, alpha=0.7,
                     color='#E8827C', label='Pre-Treatment')
    training_top = [p + t for p, t in zip(pre_counts, training_counts)]
    ax2.fill_between(months_list, pre_counts, training_top,
                     alpha=0.7, color='#999999', label='In Training / Adoption Window')
    total_top = [p + t + po for p, t, po in zip(pre_counts, training_counts, post_counts)]
    ax2.fill_between(months_list, training_top, total_top,
                     alpha=0.7, color='#4A90E2', label='Post-Adoption')

    ax2.set_xlabel('Calendar Month', fontweight='bold')
    ax2.set_ylabel('Number of Treatment Locations', fontweight='bold')
    ax2.set_title('Treatment group composition shifts over time', fontweight='bold')
    ax2.legend(loc='center left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Why a Single Cutoff Doesn\'t Work', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_event_time_normalization(data, output_path):
    """
    Three-panel figure showing transformation from calendar time to event time.
    Panel 1: Calendar time with adoption stars
    Panel 2: Calendar time with colored zone overlays
    Panel 3: Event time (normalized)
    """
    setup_plot_style()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Select 3 treatment locations for cleaner readability
    treatment_locs = data[data['treatment'] == 1]['location_id'].unique()[:3]
    colors_locs = ['#4A90E2', '#E8827C', '#77DD77']

    # --- PANEL 1: Calendar Time ---
    for i, loc_id in enumerate(treatment_locs):
        loc_data = data[data['location_id'] == loc_id].sort_values('month')
        ax1.plot(loc_data['month'], loc_data['metric'], marker='o', alpha=0.7,
                linewidth=2, label=f'Loc {loc_id}', color=colors_locs[i], markersize=4)

        # Mark adoption point
        first_comp = loc_data['first_completion'].iloc[0]
        adoption_metric = loc_data[loc_data['month'] == first_comp]['metric'].values
        if len(adoption_metric) > 0:
            ax1.scatter(first_comp, adoption_metric[0], s=100, marker='*',
                       color=colors_locs[i], edgecolors='black', linewidths=1.5, zorder=10)

    ax1.set_xlabel('Calendar Month', fontweight='bold')
    ax1.set_ylabel('Metric', fontweight='bold')
    ax1.set_title('Calendar Time', fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- PANEL 2: Mark the Zones ---
    for i, loc_id in enumerate(treatment_locs):
        loc_data = data[data['location_id'] == loc_id].sort_values('month')
        ax2.plot(loc_data['month'], loc_data['metric'], marker='o', alpha=0.7,
                linewidth=2, label=f'Loc {loc_id}', color=colors_locs[i], markersize=4)

        first_comp = loc_data['first_completion'].iloc[0]
        last_comp = loc_data['last_completion'].iloc[0]
        a_start = loc_data['adoption_start'].iloc[0]

        # Colored background zones (semi-transparent, stacked per location)
        zone_alpha = 0.08
        # Pre-treatment zone (red)
        ax2.axvspan(1, a_start, alpha=zone_alpha, color='#E8827C')
        # Adoption window zone (gray) - from adoption_start to last_completion
        ax2.axvspan(a_start, last_comp, alpha=zone_alpha, color='#999999')
        # Post-adoption zone (blue)
        ax2.axvspan(last_comp, 24, alpha=zone_alpha, color='#4A90E2')

    # Add legend for zones
    zone_legend = [
        Patch(facecolor='#E8827C', alpha=0.3, label='Pre-Treatment'),
        Patch(facecolor='#999999', alpha=0.3, label='Adoption Window'),
        Patch(facecolor='#4A90E2', alpha=0.3, label='Post-Adoption'),
    ]
    ax2.legend(handles=zone_legend, loc='upper left', fontsize=8)
    ax2.set_xlabel('Calendar Month', fontweight='bold')
    ax2.set_ylabel('Metric', fontweight='bold')
    ax2.set_title('Mark the Zones', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # --- PANEL 3: Normalize to Event Time ---
    for i, loc_id in enumerate(treatment_locs):
        loc_data = data[data['location_id'] == loc_id].sort_values('month').copy()
        first_comp = loc_data['first_completion'].iloc[0]

        # Create event time (relative to adoption)
        loc_data['event_time'] = loc_data['month'] - first_comp
        loc_data_filtered = loc_data[loc_data['event_time'].between(-6, 8)]

        ax3.plot(loc_data_filtered['event_time'], loc_data_filtered['metric'],
                marker='o', alpha=0.7, linewidth=2, label=f'Loc {loc_id}',
                color=colors_locs[i], markersize=4)

        # Mark Period 0 (adoption window)
        adoption_data = loc_data_filtered[loc_data_filtered['event_time'] == 0]
        if len(adoption_data) > 0:
            ax3.scatter(0, adoption_data['metric'].values[0], s=100, marker='*',
                       color=colors_locs[i], edgecolors='black', linewidths=1.5, zorder=10)

    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Period 0 (Adoption)')
    ax3.fill_between([-0.5, 0.5], 45, 68, alpha=0.1, color='green')
    ax3.set_xlabel('Event Time (Relative Periods)', fontweight='bold')
    ax3.set_ylabel('Metric', fontweight='bold')
    ax3.set_title('Normalize to Event Time', fontweight='bold')
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.suptitle('Normalizing to Event Time', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_event_study_results(coefficients, ci_lower, ci_upper, output_path):
    """
    Two-panel event-study plot.
    Left: Shows adoption window months as individual grayed-out points
    Right: Classic event-study with Period 0 as a single green band
    """
    setup_plot_style()

    periods = sorted(coefficients.keys())
    coefs = [coefficients[p] for p in periods]
    lower = [ci_lower[p] for p in periods]
    upper = [ci_upper[p] for p in periods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- LEFT PANEL: Including Adoption Window Months ---
    pre_periods = [p for p in periods if p < 0]
    post_periods = [p for p in periods if p > 0]
    pre_coefs = [coefficients[p] for p in pre_periods]
    post_coefs = [coefficients[p] for p in post_periods]
    pre_lower = [ci_lower[p] for p in pre_periods]
    pre_upper = [ci_upper[p] for p in pre_periods]
    post_lower = [ci_lower[p] for p in post_periods]
    post_upper = [ci_upper[p] for p in post_periods]

    # Plot pre-period normally
    ax1.fill_between(pre_periods, pre_lower, pre_upper, alpha=0.25, color='#4A90E2')
    ax1.plot(pre_periods, pre_coefs, marker='o', linewidth=2.5, markersize=8,
            color='#4A90E2', zorder=5)

    # Plot post-period normally
    ax1.fill_between(post_periods, post_lower, post_upper, alpha=0.25, color='#4A90E2')
    ax1.plot(post_periods, post_coefs, marker='o', linewidth=2.5, markersize=8,
            color='#4A90E2', zorder=5)

    # Show 6 individual adoption window months as gray hollow markers
    adoption_months_x = np.linspace(-0.4, 0.4, 6)
    for x_pos in adoption_months_x:
        ax1.scatter(x_pos, 0, s=60, marker='o', facecolors='none',
                   edgecolors='#999999', linewidths=1.5, zorder=6, alpha=0.7)

    # Shade adoption window
    ax1.axvspan(-0.5, 0.5, alpha=0.15, color='#999999')

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.axvspan(-6.5, -0.5, alpha=0.05, color='gray')

    ax1.set_xlabel('Relative Time Period', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Treatment Effect (Coefficient)', fontsize=11, fontweight='bold')
    ax1.set_title('Including Adoption Window Months', fontweight='bold')
    ax1.set_xticks(periods)
    ax1.set_xticklabels([str(p) if p != 0 else '0' for p in periods], rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add annotation for gray dots
    ax1.annotate('6 adoption months\n(excluded from estimate)',
                xy=(0, 0), xytext=(2.5, -2.5),
                fontsize=9, fontstyle='italic', color='#666666',
                arrowprops=dict(arrowstyle='->', color='#999999', lw=1),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', alpha=0.8))

    # --- RIGHT PANEL: Event-Study Coefficients (classic) ---
    ax2.fill_between(periods, lower, upper, alpha=0.25, color='#4A90E2', label='95% CI')
    ax2.plot(periods, coefs, marker='o', linewidth=2.5, markersize=8,
            color='#4A90E2', label='Point Estimate', zorder=5)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.axvspan(-0.5, 0.5, alpha=0.1, color='green', label='Adoption Window (Period 0)')
    ax2.axvspan(-6.5, -0.5, alpha=0.05, color='gray')

    ax2.set_xlabel('Relative Time Period', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Treatment Effect (Coefficient)', fontsize=11, fontweight='bold')
    ax2.set_title('Event-Study Coefficients', fontweight='bold')
    ax2.set_xticks(periods)
    ax2.set_xticklabels([str(p) if p != 0 else 'Period 0' for p in periods], rotation=45)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Match y-axes
    all_vals = lower + upper
    y_min = min(all_vals) - 0.5
    y_max = max(all_vals) + 0.5
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    fig.suptitle('Event-Study Estimates: Treatment Effect by Relative Period',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_parallel_trends(data, output_path):
    """
    Plot treatment vs control group averages over time.
    Show parallel pre-trends and post-treatment divergence.
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Aggregate by treatment and month
    agg_data = data.groupby(['month', 'treatment'])['metric'].mean().reset_index()

    treatment_data = agg_data[agg_data['treatment'] == 1]
    control_data = agg_data[agg_data['treatment'] == 0]

    ax.plot(treatment_data['month'], treatment_data['metric'],
           marker='o', linewidth=3, markersize=7, color='#4A90E2', label='Treatment Group', zorder=5)
    ax.plot(control_data['month'], control_data['metric'],
           marker='s', linewidth=3, markersize=7, color='#999999', label='Control Group', zorder=5)

    # Estimate median adoption time from treatment group
    treatment_locs = data[data['treatment'] == 1]
    median_adoption = treatment_locs['first_completion'].median()

    # Shade adoption window region
    ax.axvspan(median_adoption - 2, median_adoption + 2, alpha=0.15, color='green',
              label='Typical Adoption Window')

    # Add vertical line at median adoption
    ax.axvline(x=median_adoption, color='green', linestyle='--', alpha=0.5, linewidth=2)

    # Pre-trend annotation
    ax.text(6, 51, 'Pre-Treatment\n(Parallel Trends)', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='#FFE6E6', alpha=0.7),
           ha='center')

    # Post-treatment annotation
    ax.text(20, 56, 'Post-Treatment\n(Treatment Effect)', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='#E6F3FF', alpha=0.7),
           ha='center')

    ax.set_xlabel('Calendar Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Metric Value', fontsize=12, fontweight='bold')
    ax.set_title('Parallel Pre-Trends Between Treatment and Control',
                fontsize=13, fontweight='bold', pad=15)

    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(40, 65)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

# ============================================================================
# 5. RESULTS ANALYSIS & BENCHMARKING
# ============================================================================

def compute_did_summary(data):
    """
    Compute a simple DiD summary table: pre/post averages for treatment and control.
    Uses event-time periods: pre = periods -6 to -1, post = periods 1 to 6.

    Returns dict with all key numbers for the blog post.
    """
    analysis = data[data['period'].notna() & (data['period'] != 0)].copy()
    analysis['post'] = (analysis['period'] > 0).astype(int)

    summary = analysis.groupby(['treatment', 'post'])['metric'].mean().unstack()
    summary.columns = ['Pre-Period Avg', 'Post-Period Avg']
    summary['Change'] = summary['Post-Period Avg'] - summary['Pre-Period Avg']
    summary.index = ['Control', 'Treatment']

    did_estimate = summary.loc['Treatment', 'Change'] - summary.loc['Control', 'Change']

    print("\n" + "="*60)
    print("DIFFERENCE-IN-DIFFERENCES SUMMARY")
    print("="*60)
    print(summary.round(2).to_string())
    print(f"\nSimple DiD Estimate: {did_estimate:.2f}")
    print(f"  = Treatment Change ({summary.loc['Treatment', 'Change']:.2f})")
    print(f"  - Control Change   ({summary.loc['Control', 'Change']:.2f})")

    return {
        'summary_table': summary,
        'did_estimate': did_estimate,
        'treatment_pre': summary.loc['Treatment', 'Pre-Period Avg'],
        'treatment_post': summary.loc['Treatment', 'Post-Period Avg'],
        'control_pre': summary.loc['Control', 'Pre-Period Avg'],
        'control_post': summary.loc['Control', 'Post-Period Avg'],
    }


def run_naive_cutoff_did(data):
    """
    Naive approach: pick a single calendar cutoff (median adoption month)
    and run a simple before/after comparison.

    This is what most people try first. It underestimates because of
    composition bias -- at any given month, the treatment group is a mix
    of locations at different adoption stages.
    """
    # Use median first_completion as the cutoff
    median_cutoff = int(data[data['treatment'] == 1]['first_completion'].median())

    # Simple pre/post split at calendar cutoff
    pre = data[data['month'] < median_cutoff]
    post = data[data['month'] >= median_cutoff]

    treatment_change = (post[post['treatment'] == 1]['metric'].mean() -
                       pre[pre['treatment'] == 1]['metric'].mean())
    control_change = (post[post['treatment'] == 0]['metric'].mean() -
                     pre[pre['treatment'] == 0]['metric'].mean())

    naive_did = treatment_change - control_change

    print(f"\nNaive Calendar Cutoff DiD (cutoff = month {median_cutoff}):")
    print(f"  Treatment change: {treatment_change:.2f}")
    print(f"  Control change:   {control_change:.2f}")
    print(f"  Naive DiD:        {naive_did:.2f}")

    return naive_did


def run_naive_twfe(data):
    """
    Standard two-way fixed effects regression with individual treatment timing.
    Each treatment location's 'post' indicator uses its own first_completion date.
    Single treatment coefficient (post x treatment), location FE, time FE.

    This is what Callaway & Sant'Anna showed can be biased under staggered timing
    because already-treated locations serve as implicit controls for later adopters.
    """
    median_cutoff = int(data[data['treatment'] == 1]['first_completion'].median())

    reg_data = data.copy()

    # Key difference from naive: each treatment location uses its OWN adoption date
    reg_data['post'] = 0
    treat_mask = reg_data['treatment'] == 1
    reg_data.loc[treat_mask, 'post'] = (
        reg_data.loc[treat_mask, 'month'] >= reg_data.loc[treat_mask, 'first_completion']
    ).astype(int)
    # Control locations use median cutoff (they have no treatment date)
    ctrl_mask = reg_data['treatment'] == 0
    reg_data.loc[ctrl_mask, 'post'] = (
        reg_data.loc[ctrl_mask, 'month'] >= median_cutoff
    ).astype(int)

    reg_data['treat_post'] = reg_data['treatment'] * reg_data['post']

    # Fixed effects
    location_dummies = pd.get_dummies(reg_data['location_id'], prefix='loc', drop_first=True)
    month_dummies = pd.get_dummies(reg_data['month'], prefix='month', drop_first=True)

    Y = reg_data['metric'].astype(float)
    X = pd.concat([
        reg_data[['treat_post']].astype(float),
        location_dummies.astype(float),
        month_dummies.astype(float)
    ], axis=1)
    X = sm.add_constant(X, has_constant='add')

    model = sm.OLS(Y, X).fit()

    twfe_estimate = model.params['treat_post']
    twfe_ci = model.conf_int().loc['treat_post']

    print(f"\nStandard TWFE Estimate: {twfe_estimate:.2f}")
    print(f"  95% CI: [{twfe_ci[0]:.2f}, {twfe_ci[1]:.2f}]")

    return twfe_estimate


def get_event_study_avg_effect(coefficients):
    """Extract the average post-period treatment effect from event-study coefficients."""
    post_coefs = [coefficients[k] for k in coefficients if k > 0]
    avg_effect = np.mean(post_coefs) if post_coefs else 0
    return avg_effect


def plot_treatment_vs_control(data, output_path):
    """
    Two-panel figure showing treatment vs control group averages.
    Left: Calendar time (raw trajectories, shows where lines diverge)
    Right: Event time (normalized, shows the clean gap = treatment effect)
    """
    setup_plot_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- LEFT PANEL: Calendar Time ---
    agg_cal = data.groupby(['month', 'treatment'])['metric'].mean().reset_index()

    treat_cal = agg_cal[agg_cal['treatment'] == 1]
    ctrl_cal = agg_cal[agg_cal['treatment'] == 0]

    ax1.plot(treat_cal['month'], treat_cal['metric'], marker='o', linewidth=2.5,
            markersize=6, color='#4A90E2', label='Treatment', zorder=5)
    ax1.plot(ctrl_cal['month'], ctrl_cal['metric'], marker='s', linewidth=2.5,
            markersize=6, color='#999999', label='Control', zorder=5)

    median_adoption = int(data[data['treatment'] == 1]['first_completion'].median())
    ax1.axvline(x=median_adoption, color='green', linestyle='--', alpha=0.5,
               linewidth=1.5, label='Median Adoption')

    # Shade the gap in post-period
    post_months = treat_cal[treat_cal['month'] >= median_adoption]['month'].values
    post_treat = treat_cal[treat_cal['month'] >= median_adoption]['metric'].values
    post_ctrl = ctrl_cal[ctrl_cal['month'] >= median_adoption]['metric'].values
    if len(post_treat) == len(post_ctrl):
        ax1.fill_between(post_months, post_ctrl, post_treat, alpha=0.15, color='#4A90E2',
                         label='Treatment Effect')

    ax1.set_xlabel('Calendar Month', fontweight='bold')
    ax1.set_ylabel('Average Metric', fontweight='bold')
    ax1.set_title('Calendar Time', fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- RIGHT PANEL: Event Time ---
    event_data = data[data['period'].notna() & (data['period'] >= -6) & (data['period'] <= 6)].copy()
    agg_event = event_data.groupby(['period', 'treatment'])['metric'].mean().reset_index()

    treat_ev = agg_event[agg_event['treatment'] == 1].sort_values('period')
    ctrl_ev = agg_event[agg_event['treatment'] == 0].sort_values('period')

    ax2.plot(treat_ev['period'], treat_ev['metric'], marker='o', linewidth=2.5,
            markersize=6, color='#4A90E2', label='Treatment', zorder=5)
    ax2.plot(ctrl_ev['period'], ctrl_ev['metric'], marker='s', linewidth=2.5,
            markersize=6, color='#999999', label='Control', zorder=5)

    ax2.axvline(x=0, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    ax2.axvspan(-0.5, 0.5, alpha=0.1, color='green', label='Period 0')

    # Shade the gap in post-period
    post_ev_periods = treat_ev[treat_ev['period'] > 0]['period'].values
    post_ev_treat = treat_ev[treat_ev['period'] > 0]['metric'].values
    post_ev_ctrl = ctrl_ev[ctrl_ev['period'] > 0]['metric'].values
    if len(post_ev_treat) == len(post_ev_ctrl):
        ax2.fill_between(post_ev_periods, post_ev_ctrl, post_ev_treat, alpha=0.15,
                         color='#4A90E2', label='Treatment Effect')

    ax2.set_xlabel('Relative Period (Event Time)', fontweight='bold')
    ax2.set_ylabel('Average Metric', fontweight='bold')
    ax2.set_title('Event Time (Normalized)', fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Treatment vs. Control: Measuring the Lift', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_control_comparison(data, output_path):
    """
    Mini-example showing how treatment locations at different event times
    are compared to the control group at the SAME calendar month.
    Illustrates the concurrent calendar-time comparison.
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(12, 7))

    # Pick 3 treatment locations with different adoption timings
    treat_locs = data[data['treatment'] == 1]
    unique_locs = treat_locs.groupby('location_id')['first_completion'].first()
    # Find locations with early, mid, and late adoption
    sorted_locs = unique_locs.sort_values()
    early_loc = sorted_locs.iloc[int(len(sorted_locs) * 0.15)]
    mid_loc = sorted_locs.iloc[int(len(sorted_locs) * 0.5)]
    late_loc = sorted_locs.iloc[int(len(sorted_locs) * 0.85)]

    early_id = sorted_locs[sorted_locs == early_loc].index[0]
    mid_id = sorted_locs[sorted_locs == mid_loc].index[0]
    late_id = sorted_locs[sorted_locs == late_loc].index[0]

    example_locs = [(early_id, '#4A90E2', 'Location A'),
                    (mid_id, '#77DD77', 'Location B'),
                    (late_id, '#E8827C', 'Location C')]

    # Plot control group average
    ctrl_avg = data[data['treatment'] == 0].groupby('month')['metric'].mean()
    ax.plot(ctrl_avg.index, ctrl_avg.values, linewidth=3, color='#999999',
            label='Control Group Average', zorder=3, marker='s', markersize=5)

    # Plot each example treatment location
    for loc_id, color, label in example_locs:
        loc_data = data[data['location_id'] == loc_id].sort_values('month')
        fc = int(loc_data['first_completion'].iloc[0])

        ax.plot(loc_data['month'], loc_data['metric'], linewidth=2, color=color,
                label=f'{label} (adopted Month {fc})', alpha=0.8, marker='o', markersize=4)

        # Draw arrows from treatment location's post-period to control at same month
        # Show 2 example comparisons per location (at fc+1 and fc+3)
        for offset in [1, 3]:
            compare_month = fc + offset
            if compare_month > 24:
                continue
            treat_val = loc_data[loc_data['month'] == compare_month]['metric'].values
            ctrl_val = ctrl_avg.get(compare_month)
            if len(treat_val) > 0 and ctrl_val is not None:
                # Vertical arrow between treatment and control
                ax.annotate('', xy=(compare_month, treat_val[0]),
                           xytext=(compare_month, ctrl_val),
                           arrowprops=dict(arrowstyle='<->', color=color,
                                          lw=1.5, alpha=0.6))

        # Mark adoption point
        adoption_val = loc_data[loc_data['month'] == fc]['metric'].values
        if len(adoption_val) > 0:
            ax.scatter(fc, adoption_val[0], s=120, marker='*', color=color,
                      edgecolors='black', linewidths=1, zorder=10)

    # Add annotation explaining the arrows
    ax.annotate('Arrows = DiD comparison\nat the same calendar month',
               xy=(20, ctrl_avg.iloc[-5] - 1),
               fontsize=10, fontstyle='italic', color='#555555',
               ha='center',
               bbox=dict(boxstyle='round,pad=0.4',
                        facecolor='#E8F4FD', alpha=0.9))

    ax.set_xlabel('Calendar Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    ax.set_title('How Treatment Compares to Control: Same Calendar Month, Different Adoption Times',
                fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_method_comparison(naive_cutoff, twfe_estimate, event_study_avg, true_effect, output_path):
    """
    Bar chart comparing the three estimation approaches against the true effect.
    """
    setup_plot_style()

    methods = ['Naive Calendar\nCutoff', 'Standard\nTWFE', 'Event-Study\nDesign']
    estimates = [naive_cutoff, twfe_estimate, event_study_avg]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#E8827C', '#FFB347', '#77DD77']
    bars = ax.bar(methods, estimates, color=colors, edgecolor='black', alpha=0.85, width=0.5)

    # True effect line
    ax.axhline(y=true_effect, color='black', linestyle='--', linewidth=2,
              label=f'True Effect ({true_effect:.1f})', zorder=10)

    # Add value labels
    for bar, est in zip(bars, estimates):
        bias_pct = ((est - true_effect) / true_effect) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{est:.2f}\n({bias_pct:+.0f}% bias)', ha='center', va='bottom', fontsize=10,
               fontweight='bold')

    ax.set_ylabel('Estimated Treatment Effect', fontsize=12, fontweight='bold')
    ax.set_title('How Much Does the Approach Matter?', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(estimates + [true_effect]) * 1.35])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_method_lift_comparison(data, output_path):
    """
    Three-panel figure (stacked vertically) showing treatment vs control
    for each methodology. Each panel gets full width for readability.
    """
    setup_plot_style()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16))
    fig.subplots_adjust(hspace=0.35)

    median_cutoff = int(data[data['treatment'] == 1]['first_completion'].median())

    # Aggregate by month and treatment
    agg_cal = data.groupby(['month', 'treatment'])['metric'].mean().reset_index()
    treat_cal = agg_cal[agg_cal['treatment'] == 1]
    ctrl_cal = agg_cal[agg_cal['treatment'] == 0]

    # --- Panel A: Naive Calendar Cutoff ---
    ax1.plot(treat_cal['month'], treat_cal['metric'], marker='o', linewidth=2.5,
            markersize=6, color='#4A90E2', label='Treatment (all locations pooled)')
    ax1.plot(ctrl_cal['month'], ctrl_cal['metric'], marker='s', linewidth=2.5,
            markersize=6, color='#999999', label='Control')
    ax1.axvline(x=median_cutoff, color='red', linestyle='--', alpha=0.7,
               linewidth=2, label=f'Single Cutoff (Month {median_cutoff})')
    ax1.set_xlabel('Calendar Month', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Average Metric', fontsize=11, fontweight='bold')
    ax1.set_title('A: Naive Calendar Cutoff -- One line for all treatment locations, one arbitrary split',
                 fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- Panel B: TWFE (split early vs late adopters) ---
    treat_locs_data = data[data['treatment'] == 1]
    median_fc = treat_locs_data['first_completion'].median()

    early_ids = treat_locs_data[
        treat_locs_data['first_completion'] <= median_fc
    ]['location_id'].unique()
    late_ids = treat_locs_data[
        treat_locs_data['first_completion'] > median_fc
    ]['location_id'].unique()

    early_agg = data[data['location_id'].isin(early_ids)].groupby(
        'month')['metric'].mean()
    late_agg = data[data['location_id'].isin(late_ids)].groupby(
        'month')['metric'].mean()

    ax2.plot(ctrl_cal['month'], ctrl_cal['metric'], marker='s',
            linewidth=2.5, markersize=6, color='#999999', label='Control')
    ax2.plot(early_agg.index, early_agg.values, marker='o',
            linewidth=2.5, markersize=6, color='#4A90E2',
            label='Early Adopters (adopted before median)')
    ax2.plot(late_agg.index, late_agg.values, marker='^',
            linewidth=2.5, markersize=6, color='#E8827C',
            label='Late Adopters (adopted after median)')

    # Mark the early and late median adoption points
    early_med = int(data[data['location_id'].isin(early_ids)][
        'first_completion'].median())
    late_med = int(data[data['location_id'].isin(late_ids)][
        'first_completion'].median())
    ax2.axvline(x=early_med, color='#4A90E2', linestyle='--',
               alpha=0.5, linewidth=1.5, label=f'Early median (Month {early_med})')
    ax2.axvline(x=late_med, color='#E8827C', linestyle='--',
               alpha=0.5, linewidth=1.5, label=f'Late median (Month {late_med})')

    # Shade the problematic region where early adopters act as controls
    ax2.axvspan(early_med, late_med, alpha=0.08, color='orange')
    ax2.annotate('Early adopters already treated here,\nbut TWFE uses them as implicit\ncomparisons for late adopters',
                xy=((early_med + late_med) / 2, ctrl_cal['metric'].iloc[0]),
                fontsize=9, fontstyle='italic', color='#555555',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.4',
                         facecolor='#FFF3CD', alpha=0.9))

    ax2.set_xlabel('Calendar Month', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Average Metric', fontsize=11, fontweight='bold')
    ax2.set_title('B: Standard TWFE -- Uses individual timing, but pools early and late adopters into one estimate',
                 fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- Panel C: Event-Study (event time) ---
    event_data = data[data['period'].notna() & (data['period'] >= -6) & (data['period'] <= 6)].copy()
    agg_event = event_data.groupby(['period', 'treatment'])['metric'].mean().reset_index()
    treat_ev = agg_event[agg_event['treatment'] == 1].sort_values('period')
    ctrl_ev = agg_event[agg_event['treatment'] == 0].sort_values('period')

    ax3.plot(treat_ev['period'], treat_ev['metric'], marker='o', linewidth=2.5,
            markersize=6, color='#4A90E2', label='Treatment')
    ax3.plot(ctrl_ev['period'], ctrl_ev['metric'], marker='s', linewidth=2.5,
            markersize=6, color='#999999', label='Control')
    ax3.axvline(x=0, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    ax3.axvspan(-0.5, 0.5, alpha=0.1, color='green', label='Period 0 (Adoption Window)')

    # Shade the gap
    post_ev_periods = treat_ev[treat_ev['period'] > 0]['period'].values
    post_ev_treat = treat_ev[treat_ev['period'] > 0]['metric'].values
    post_ev_ctrl = ctrl_ev[ctrl_ev['period'] > 0]['metric'].values
    if len(post_ev_treat) == len(post_ev_ctrl):
        ax3.fill_between(post_ev_periods, post_ev_ctrl, post_ev_treat, alpha=0.15,
                         color='#4A90E2', label='Treatment Effect')
    ax3.set_xlabel('Relative Period (Event Time)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Average Metric', fontsize=11, fontweight='bold')
    ax3.set_title('C: Event-Study Design -- Each location aligned to its own adoption, effect measured in relative time',
                 fontsize=12, fontweight='bold', pad=10)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)

    fig.suptitle('Same Data, Three Approaches', fontsize=15, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================================
# 6. SCENARIO B: EVERYONE EVENTUALLY TREATED
# ============================================================================

def generate_scenario_b_data(n_locations=200, n_months=30,
                              adoption_range=(3, 24)):
    """
    All locations eventually get treated. No permanent control group.
    Adoption starts staggered across adoption_range.
    Longer timeline (30 months) so late adopters have enough post-data.

    Same training mechanics and treatment effect as Scenario A.
    """
    np.random.seed(42)
    records = []

    for loc_id in range(1, n_locations + 1):
        adoption_start = np.random.randint(
            adoption_range[0], adoption_range[1] + 1
        )
        n_employees = np.random.randint(1, 6)

        if np.random.random() < 0.4:
            employee_starts = [adoption_start] * n_employees
        else:
            spread = np.random.randint(1, 4)
            employee_starts = [
                adoption_start + np.random.randint(0, spread)
                for _ in range(n_employees)
            ]

        completions = [s + 6 for s in employee_starts]
        first_completion = min(completions)
        last_completion = max(completions)
        window = last_completion - first_completion

        if window > 6:
            continue

        for month in range(1, n_months + 1):
            baseline = 50 + 0.3 * month
            noise = np.random.normal(0, 2)

            effect = 0
            if month >= first_completion:
                months_post = min(
                    month - first_completion + 1, 3
                )
                effect = months_post * 1.5

            records.append({
                'location_id': loc_id,
                'month': month,
                'metric': baseline + effect + noise,
                'first_completion': first_completion,
                'last_completion': last_completion,
                'adoption_window': window,
                'adoption_start': adoption_start,
            })

    data = pd.DataFrame(records)
    n_locs = data['location_id'].nunique()
    print(f"Scenario B: {n_locs} locations (all treated)")
    return data


def define_cohort_controls(data, pre_periods=6, post_periods=6):
    """
    For each cohort (locations completing in the same month),
    define temporary controls: locations that haven't started
    treatment by the end of this cohort's analysis window.

    Returns a combined DataFrame with cohort-specific
    treatment/control labels and event-time periods.
    """
    all_cohort_data = []

    # Get unique first_completion months as cohorts
    cohort_months = sorted(
        data['first_completion'].dropna().unique()
    )

    for cohort_month in cohort_months:
        cohort_month = int(cohort_month)

        # Cohort = locations with first_completion == cohort_month
        cohort_ids = data[
            data['first_completion'] == cohort_month
        ]['location_id'].unique()

        if len(cohort_ids) == 0:
            continue

        # Analysis window for this cohort
        window_start = cohort_month - pre_periods
        window_end = cohort_month + post_periods

        # Control = locations whose adoption_start > window_end
        # (haven't even started training by end of analysis window)
        control_ids = data[
            data['adoption_start'] > window_end
        ]['location_id'].unique()

        if len(control_ids) < 3:
            continue  # Need meaningful control group

        # Build cohort dataset
        for loc_id in cohort_ids:
            loc_data = data[
                (data['location_id'] == loc_id)
                & (data['month'] >= window_start)
                & (data['month'] <= window_end)
            ].copy()
            loc_data['cohort'] = cohort_month
            loc_data['cohort_treatment'] = 1
            fc = loc_data['first_completion'].iloc[0]
            loc_data['event_time'] = loc_data['month'] - fc
            loc_data['period'] = np.nan
            in_range = loc_data['event_time'].between(
                -pre_periods, post_periods
            )
            loc_data.loc[in_range, 'period'] = (
                loc_data.loc[in_range, 'event_time'].astype(int)
            )
            all_cohort_data.append(loc_data)

        for loc_id in control_ids:
            loc_data = data[
                (data['location_id'] == loc_id)
                & (data['month'] >= window_start)
                & (data['month'] <= window_end)
            ].copy()
            loc_data['cohort'] = cohort_month
            loc_data['cohort_treatment'] = 0
            # Align control to cohort's event time
            loc_data['event_time'] = (
                loc_data['month'] - cohort_month
            )
            loc_data['period'] = np.nan
            in_range = loc_data['event_time'].between(
                -pre_periods, post_periods
            )
            loc_data.loc[in_range, 'period'] = (
                loc_data.loc[in_range, 'event_time'].astype(int)
            )
            all_cohort_data.append(loc_data)

    if not all_cohort_data:
        print("Warning: no valid cohorts found")
        return pd.DataFrame()

    combined = pd.concat(all_cohort_data, ignore_index=True)

    n_cohorts = combined['cohort'].nunique()
    n_treat = combined[
        combined['cohort_treatment'] == 1
    ]['location_id'].nunique()
    n_ctrl = combined[
        combined['cohort_treatment'] == 0
    ]['location_id'].nunique()
    print(f"Cohort analysis: {n_cohorts} cohorts, "
          f"{n_treat} treated locs, {n_ctrl} control locs")

    return combined


def run_scenario_b_event_study(cohort_data, pre_periods=6,
                                post_periods=6):
    """
    Run event-study regression on cohort-defined data.
    Uses cohort_treatment (not 'treatment') for the interaction terms.
    Includes cohort fixed effects alongside location and time FE.
    """
    reg_data = cohort_data[
        cohort_data['period'].notna()
        & (cohort_data['period'] >= -pre_periods)
        & (cohort_data['period'] <= post_periods)
    ].copy()

    if len(reg_data) == 0:
        print("No data for Scenario B regression")
        return None, {}, {}, {}

    # Treatment x period interaction dummies
    for k in range(-pre_periods, post_periods + 1):
        reg_data[f'period_{k}'] = (
            (reg_data['period'] == k)
            & (reg_data['cohort_treatment'] == 1)
        ).astype(int)

    # Drop period -1 (reference)
    reg_data = reg_data.drop(columns=['period_-1'])

    # Fixed effects: location, month, cohort
    loc_fe = pd.get_dummies(
        reg_data['location_id'], prefix='loc', drop_first=True
    )
    time_fe = pd.get_dummies(
        reg_data['month'], prefix='month', drop_first=True
    )
    cohort_fe = pd.get_dummies(
        reg_data['cohort'], prefix='cohort', drop_first=True
    )

    period_cols = [
        f'period_{k}'
        for k in range(-pre_periods, post_periods + 1)
        if k != -1
    ]
    X = pd.concat(
        [reg_data[period_cols], loc_fe, time_fe, cohort_fe],
        axis=1
    ).astype(float)
    X = sm.add_constant(X, has_constant='add')
    Y = reg_data['metric'].astype(float)

    results = sm.OLS(Y, X).fit()

    coefs, ci_lo, ci_hi = {}, {}, {}
    for k in range(-pre_periods, post_periods + 1):
        if k == -1:
            coefs[k], ci_lo[k], ci_hi[k] = 0, 0, 0
            continue
        col = f'period_{k}'
        if col in results.params.index:
            coefs[k] = results.params[col]
            ci = results.conf_int().loc[col]
            ci_lo[k], ci_hi[k] = ci[0], ci[1]
        else:
            coefs[k], ci_lo[k], ci_hi[k] = 0, 0, 0

    return results, coefs, ci_lo, ci_hi


def compute_scenario_b_did(cohort_data):
    """
    Simple DiD table for Scenario B using cohort-defined controls.
    Pre = periods -6 to -1, Post = +1 to +6.
    """
    analysis = cohort_data[
        cohort_data['period'].notna()
        & (cohort_data['period'] != 0)
    ].copy()
    analysis['post'] = (analysis['period'] > 0).astype(int)

    summary = analysis.groupby(
        ['cohort_treatment', 'post']
    )['metric'].mean().unstack()
    summary.columns = ['Pre-Period Avg', 'Post-Period Avg']
    summary['Change'] = (
        summary['Post-Period Avg'] - summary['Pre-Period Avg']
    )
    summary.index = ['Control', 'Treatment']

    did = (
        summary.loc['Treatment', 'Change']
        - summary.loc['Control', 'Change']
    )

    print(f"\nScenario B DiD Estimate: {did:.2f}")
    print(summary.round(2).to_string())
    return summary, did


def plot_scenario_b_event_study(coefs_b, ci_lo_b, ci_hi_b,
                                 output_path):
    """Event-study plot for Scenario B (cohort-based controls)."""
    setup_plot_style()

    periods = sorted(coefs_b.keys())
    coefs_list = [coefs_b[p] for p in periods]
    lower = [ci_lo_b[p] for p in periods]
    upper = [ci_hi_b[p] for p in periods]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(periods, lower, upper, alpha=0.25,
                    color='#E8827C', label='95% CI')
    ax.plot(periods, coefs_list, marker='o', linewidth=2.5,
            markersize=8, color='#E8827C',
            label='Point Estimate', zorder=5)

    ax.axhline(y=0, color='black', linestyle='-',
               linewidth=1, alpha=0.5)
    ax.axvspan(-0.5, 0.5, alpha=0.1, color='green',
               label='Adoption Window (Period 0)')
    ax.axvspan(-6.5, -0.5, alpha=0.05, color='gray')

    ax.set_xlabel('Relative Time Period', fontsize=12,
                  fontweight='bold')
    ax.set_ylabel('Treatment Effect (Coefficient)', fontsize=12,
                  fontweight='bold')
    ax.set_title(
        'Scenario B: Event-Study Estimates (Cohort Controls)',
        fontsize=13, fontweight='bold', pad=15
    )
    ax.set_xticks(periods)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_scenario_b_treatment_vs_control(cohort_data, output_path):
    """
    Treatment vs control in event time for Scenario B.
    Shows the lift using cohort-defined temporary controls.
    """
    setup_plot_style()

    event_data = cohort_data[
        cohort_data['period'].notna()
        & (cohort_data['period'] >= -6)
        & (cohort_data['period'] <= 6)
    ].copy()

    agg = event_data.groupby(
        ['period', 'cohort_treatment']
    )['metric'].mean().reset_index()

    treat = agg[agg['cohort_treatment'] == 1].sort_values('period')
    ctrl = agg[agg['cohort_treatment'] == 0].sort_values('period')

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(treat['period'], treat['metric'], marker='o',
            linewidth=2.5, markersize=6, color='#E8827C',
            label='Treatment (Cohort)', zorder=5)
    ax.plot(ctrl['period'], ctrl['metric'], marker='s',
            linewidth=2.5, markersize=6, color='#999999',
            label='Control (Not-Yet-Treated)', zorder=5)

    ax.axvline(x=0, color='green', linestyle='--', alpha=0.5,
               linewidth=1.5)
    ax.axvspan(-0.5, 0.5, alpha=0.1, color='green',
               label='Period 0')

    # Shade treatment effect gap
    post_p = treat[treat['period'] > 0]['period'].values
    post_t = treat[treat['period'] > 0]['metric'].values
    post_c = ctrl[ctrl['period'] > 0]['metric'].values
    if len(post_t) == len(post_c):
        ax.fill_between(post_p, post_c, post_t, alpha=0.15,
                         color='#E8827C', label='Treatment Effect')

    ax.set_xlabel('Relative Period (Event Time)', fontsize=12,
                  fontweight='bold')
    ax.set_ylabel('Average Metric', fontsize=12, fontweight='bold')
    ax.set_title(
        'Scenario B: Treatment vs. Temporary Controls',
        fontsize=13, fontweight='bold', pad=15
    )
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_scenario_comparison(coefs_a, ci_lo_a, ci_hi_a,
                              coefs_b, ci_lo_b, ci_hi_b,
                              output_path):
    """
    Side-by-side event-study plots: Scenario A vs Scenario B.
    Includes true effect reference line at y=4.5.
    """
    setup_plot_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    periods_a = sorted(coefs_a.keys())
    periods_b = sorted(coefs_b.keys())

    true_effect = 4.5

    # Scenario A
    ax1.fill_between(
        periods_a,
        [ci_lo_a[p] for p in periods_a],
        [ci_hi_a[p] for p in periods_a],
        alpha=0.25, color='#4A90E2'
    )
    ax1.plot(
        periods_a, [coefs_a[p] for p in periods_a],
        marker='o', linewidth=2.5, markersize=8,
        color='#4A90E2', zorder=5
    )
    ax1.axhline(y=0, color='black', linestyle='-',
                linewidth=1, alpha=0.5)
    ax1.axhline(y=true_effect, color='black', linestyle='--',
                linewidth=1.5, alpha=0.7, label=f'True Effect ({true_effect})')
    ax1.axvspan(-0.5, 0.5, alpha=0.1, color='green')
    ax1.set_xlabel('Relative Period', fontweight='bold')
    ax1.set_ylabel('Treatment Effect', fontweight='bold')
    ax1.set_title('Scenario A: Clean Holdout',
                  fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # Scenario B
    ax2.fill_between(
        periods_b,
        [ci_lo_b[p] for p in periods_b],
        [ci_hi_b[p] for p in periods_b],
        alpha=0.25, color='#E8827C'
    )
    ax2.plot(
        periods_b, [coefs_b[p] for p in periods_b],
        marker='o', linewidth=2.5, markersize=8,
        color='#E8827C', zorder=5
    )
    ax2.axhline(y=0, color='black', linestyle='-',
                linewidth=1, alpha=0.5)
    ax2.axhline(y=true_effect, color='black', linestyle='--',
                linewidth=1.5, alpha=0.7, label=f'True Effect ({true_effect})')
    ax2.axvspan(-0.5, 0.5, alpha=0.1, color='green')
    ax2.set_xlabel('Relative Period', fontweight='bold')
    ax2.set_ylabel('Treatment Effect', fontweight='bold')
    ax2.set_title('Scenario B: Cohort Controls',
                  fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Match y-axes
    all_vals = (
        [ci_lo_a[p] for p in periods_a]
        + [ci_hi_a[p] for p in periods_a]
        + [ci_lo_b[p] for p in periods_b]
        + [ci_hi_b[p] for p in periods_b]
        + [true_effect]
    )
    y_min = min(all_vals) - 0.5
    y_max = max(all_vals) + 0.5
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    fig.suptitle('Event-Study Estimates: Two Scenarios',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================================
# 6b. ADDITIONAL VISUALIZATION FUNCTIONS
# ============================================================================

def plot_shrinking_controls(cohort_data, output_path):
    """
    Grouped bar chart showing treatment count vs available control count
    per cohort month. As cohort month increases, controls shrink.
    """
    setup_plot_style()

    # Get unique cohort months
    cohort_months = sorted(cohort_data['cohort'].unique())

    treat_counts = []
    control_counts = []

    for cm in cohort_months:
        cm_data = cohort_data[cohort_data['cohort'] == cm]
        n_treat = cm_data[cm_data['cohort_treatment'] == 1]['location_id'].nunique()
        n_ctrl = cm_data[cm_data['cohort_treatment'] == 0]['location_id'].nunique()
        treat_counts.append(n_treat)
        control_counts.append(n_ctrl)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(cohort_months))
    width = 0.35

    bars_treat = ax.bar(x - width/2, treat_counts, width,
                        color='#4A90E2', label='Treatment Locations',
                        edgecolor='black', alpha=0.85)
    bars_ctrl = ax.bar(x + width/2, control_counts, width,
                       color='#999999', label='Available Control Locations',
                       edgecolor='black', alpha=0.85)

    ax.set_xlabel('Cohort Month (First Completion)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Locations', fontsize=12, fontweight='bold')
    ax.set_title('Shrinking Control Pool: Later Cohorts Have Fewer Controls',
                fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(cm)) for cm in cohort_months], rotation=45)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_raw_metric_lift(data, output_path):
    """
    Plot actual average metric values by event-time period for
    treatment vs control groups (Scenario A).
    Shows the raw lift with filled area in post-period.
    """
    setup_plot_style()

    event_data = data[
        data['period'].notna()
        & (data['period'] >= -6)
        & (data['period'] <= 6)
    ].copy()

    agg = event_data.groupby(
        ['period', 'treatment']
    )['metric'].mean().reset_index()

    treat = agg[agg['treatment'] == 1].sort_values('period')
    ctrl = agg[agg['treatment'] == 0].sort_values('period')

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(treat['period'], treat['metric'], marker='o',
            linewidth=2.5, markersize=8, color='#4A90E2',
            label='Treatment', zorder=5)
    ax.plot(ctrl['period'], ctrl['metric'], marker='s',
            linewidth=2.5, markersize=8, color='#999999',
            label='Control', zorder=5)

    # Filled area in post-period
    post_treat = treat[treat['period'] > 0]
    post_ctrl = ctrl[ctrl['period'] > 0]
    if len(post_treat) == len(post_ctrl):
        ax.fill_between(post_treat['period'].values,
                        post_ctrl['metric'].values,
                        post_treat['metric'].values,
                        alpha=0.2, color='#4A90E2',
                        label='Treatment Effect (Lift)')

    # Green band on period 0
    ax.axvspan(-0.5, 0.5, alpha=0.1, color='green',
               label='Adoption Window (Period 0)')

    ax.set_xlabel('Relative Period (Event Time)', fontsize=12,
                  fontweight='bold')
    ax.set_ylabel('Average Metric Value', fontsize=12,
                  fontweight='bold')
    ax.set_title('Scenario A: Raw Metric Lift by Event-Time Period',
                fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_scenario_b_raw_lift(cohort_data, output_path):
    """
    Plot actual average metric values by event-time period for
    treatment vs control groups (Scenario B cohort data).
    Shows the raw lift with filled area in post-period.
    """
    setup_plot_style()

    event_data = cohort_data[
        cohort_data['period'].notna()
        & (cohort_data['period'] >= -6)
        & (cohort_data['period'] <= 6)
    ].copy()

    agg = event_data.groupby(
        ['period', 'cohort_treatment']
    )['metric'].mean().reset_index()

    treat = agg[agg['cohort_treatment'] == 1].sort_values('period')
    ctrl = agg[agg['cohort_treatment'] == 0].sort_values('period')

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(treat['period'], treat['metric'], marker='o',
            linewidth=2.5, markersize=8, color='#E8827C',
            label='Treatment (Cohort)', zorder=5)
    ax.plot(ctrl['period'], ctrl['metric'], marker='s',
            linewidth=2.5, markersize=8, color='#999999',
            label='Control (Not-Yet-Treated)', zorder=5)

    # Filled area in post-period
    post_treat = treat[treat['period'] > 0]
    post_ctrl = ctrl[ctrl['period'] > 0]
    if len(post_treat) == len(post_ctrl):
        ax.fill_between(post_treat['period'].values,
                        post_ctrl['metric'].values,
                        post_treat['metric'].values,
                        alpha=0.2, color='#E8827C',
                        label='Treatment Effect (Lift)')

    # Green band on period 0
    ax.axvspan(-0.5, 0.5, alpha=0.1, color='green',
               label='Adoption Window (Period 0)')

    ax.set_xlabel('Relative Period (Event Time)', fontsize=12,
                  fontweight='bold')
    ax.set_ylabel('Average Metric Value', fontsize=12,
                  fontweight='bold')
    ax.set_title('Scenario B: Raw Metric Lift by Event-Time Period',
                fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

def main(output_dir=None):
    """Run the complete staggered DiD simulation (both scenarios)."""
    import os

    print("="*70)
    print("STAGGERED DIFFERENCE-IN-DIFFERENCES SIMULATION")
    print("="*70)

    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(
            script_dir, '..', 'images', 'posts', 'staggered-did'
        )
        output_dir = os.path.abspath(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # ==== SCENARIO A: Clean Holdout Control ====
    print("\n" + "="*70)
    print("SCENARIO A: CLEAN HOLDOUT CONTROL")
    print("="*70)

    print("\n[A1] Generating Scenario A data...")
    data = generate_synthetic_data(
        n_locations=200, n_treatment=120,
        n_control=80, n_months=24
    )

    print("\n[A2] Normalizing to event time...")
    data = normalize_to_event_time(data)
    print(f"Data shape: {data.shape}")
    print(f"Locations: {data['location_id'].nunique()}")

    print("\n[A3] Running event-study regression...")
    results_a, coefs_a, ci_lo_a, ci_hi_a = run_event_study_regression(data)

    print("\n[A4] Computing DiD summary...")
    did_summary_a = compute_did_summary(data)

    print("\n[A5] Running benchmarking comparisons...")
    naive_cutoff = run_naive_cutoff_did(data)
    twfe_estimate = run_naive_twfe(data)
    event_study_avg_a = get_event_study_avg_effect(coefs_a)

    true_effect = 4.5

    print(f"\nScenario A: Naive={naive_cutoff:.2f}, TWFE={twfe_estimate:.2f}, Event-Study={event_study_avg_a:.2f}")

    print("\n[A6] Generating Scenario A plots...")
    plot_staggered_adoption_heatmap(data, f"{output_dir}/staggered-adoption-heatmap.png")
    plot_naive_did_problem(data, f"{output_dir}/naive-did-problem.png")
    plot_event_time_normalization(data, f"{output_dir}/event-time-normalization.png")
    plot_event_study_results(coefs_a, ci_lo_a, ci_hi_a, f"{output_dir}/event-study-plot.png")
    plot_parallel_trends(data, f"{output_dir}/parallel-trends.png")
    plot_treatment_vs_control(data, f"{output_dir}/treatment-vs-control-lift.png")
    plot_control_comparison(data, f"{output_dir}/control-comparison.png")
    plot_method_comparison(naive_cutoff, twfe_estimate, event_study_avg_a, true_effect, f"{output_dir}/method-comparison.png")
    plot_method_lift_comparison(data, f"{output_dir}/method-lift-comparison.png")
    plot_raw_metric_lift(data, os.path.join(output_dir, 'scenario-a-raw-lift.png'))



    # ==== SCENARIO B: Everyone Eventually Treated ====
    print("\n" + "="*70)
    print("SCENARIO B: EVERYONE EVENTUALLY TREATED")
    print("="*70)

    print("\n[B1] Generating Scenario B data...")
    data_b = generate_scenario_b_data(n_locations=200, n_months=30)

    print("\n[B2] Defining cohort controls...")
    cohort_data = define_cohort_controls(data_b)

    coefs_b, ci_lo_b, ci_hi_b = {}, {}, {}

    if len(cohort_data) > 0:
        print("\n[B3] Running Scenario B event-study...")
        results_b, coefs_b, ci_lo_b, ci_hi_b = run_scenario_b_event_study(cohort_data)

        print("\n[B4] Computing Scenario B DiD summary...")
        summary_b, did_b = compute_scenario_b_did(cohort_data)

        post_coefs_b = [coefs_b[k] for k in coefs_b if k > 0]
        event_study_avg_b = np.mean(post_coefs_b) if post_coefs_b else 0
        bias_b = ((event_study_avg_b - true_effect) / true_effect) * 100
        print(f"Scenario B event-study avg: {event_study_avg_b:.2f} ({bias_b:+.0f}% bias)")

        print("\n[B5] Generating Scenario B plots...")
        plot_scenario_b_event_study(coefs_b, ci_lo_b, ci_hi_b, f"{output_dir}/scenario-b-event-study.png")
        plot_scenario_b_treatment_vs_control(cohort_data, f"{output_dir}/scenario-b-treatment-vs-control.png")
        plot_scenario_comparison(coefs_a, ci_lo_a, ci_hi_a, coefs_b, ci_lo_b, ci_hi_b, f"{output_dir}/scenario-comparison.png")
        plot_shrinking_controls(cohort_data, os.path.join(output_dir, 'shrinking-controls.png'))
        plot_scenario_b_raw_lift(cohort_data, os.path.join(output_dir, 'scenario-b-raw-lift.png'))
    else:
        print("WARNING: Scenario B produced no cohort data")

    # ==== VERIFICATION ====
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)

    required_files = [
        'staggered-adoption-heatmap.png', 'naive-did-problem.png',
        'event-time-normalization.png', 'event-study-plot.png',
        'parallel-trends.png', 'treatment-vs-control-lift.png',
        'control-comparison.png', 'method-comparison.png',
        'method-lift-comparison.png',
        'scenario-b-event-study.png',
        'scenario-b-treatment-vs-control.png',
        'scenario-comparison.png',
        'shrinking-controls.png',
        'scenario-a-raw-lift.png',
        'scenario-b-raw-lift.png',
    ]

    print("\nFile generation verification:")
    for fname in required_files:
        fpath = f"{output_dir}/{fname}"
        if os.path.exists(fpath):
            file_size = os.path.getsize(fpath) / 1024
            print(f"  [OK] {fname} ({file_size:.1f} KB)")
        else:
            print(f"  [MISSING] {fname}")

    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print(f"\nAll outputs saved to: {output_dir}")

    return {
        'scenario_a': {
            'data': data, 'results': results_a,
            'coefficients': coefs_a, 'did_summary': did_summary_a,
            'naive': naive_cutoff, 'twfe': twfe_estimate,
            'event_study_avg': event_study_avg_a,
        },
        'scenario_b': {
            'data': data_b, 'cohort_data': cohort_data,
            'coefficients': coefs_b,
        },
    }


if __name__ == "__main__":
    results = main()
