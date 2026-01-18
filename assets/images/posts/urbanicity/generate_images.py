"""
Generate images for the Geospatial Clustering Urbanicity blog post.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

# Data path
DATA_PATH = r"C:\Users\Interview Prep\Desktop\Portfolio Github\porkbun-content\hospitals.csv"
OUTPUT_DIR = "."

# Custom colormap for urbanicity
URBANICITY_COLORS = {
    'Very Urban': '#7b3294',
    'Urban': '#c2a5cf',
    'Suburban': '#f7f7f7',
    'Rural': '#a6dba0',
    'Very Rural': '#008837'
}


def load_hospitals():
    """Load hospital data."""
    print("Loading hospital data...")
    df = pd.read_csv(DATA_PATH)

    # Filter to open hospitals with valid coordinates
    df = df[df['STATUS'] == 'OPEN'].copy()
    df = df[df['X'].notna() & df['Y'].notna()].copy()

    # Use X for longitude, Y for latitude (per the CSV header)
    df['Longitude'] = df['X']
    df['Latitude'] = df['Y']

    # Handle beds column
    df['BEDS'] = pd.to_numeric(df['BEDS'], errors='coerce').fillna(0)
    df.loc[df['BEDS'] < 0, 'BEDS'] = 0

    print(f"  Loaded {len(df)} open hospitals")
    return df


def calculate_distance_matrix(df, sample_size=None):
    """Calculate pairwise distances between hospitals."""
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    coords = df[['Latitude', 'Longitude']].values

    # Calculate haversine distances (approximate using euclidean on lat/lon)
    # For proper haversine, multiply by ~111km per degree
    distances = cdist(coords, coords, metric='euclidean') * 111000  # meters

    return distances, df


def classify_single_distance(df, distances):
    """Classify using single minimum distance method."""
    # Replace diagonal with inf to avoid self-distance
    distances_copy = distances.copy()
    np.fill_diagonal(distances_copy, np.inf)

    min_distances = np.min(distances_copy, axis=1)

    # Classification thresholds (in meters)
    conditions = [
        min_distances <= 3000,      # Very Urban: < 3km
        min_distances <= 12000,     # Urban: 3-12km
        min_distances <= 20000,     # Suburban: 12-20km
        min_distances <= 30000,     # Rural: 20-30km
    ]
    choices = ['Very Urban', 'Urban', 'Suburban', 'Rural']

    df = df.copy()
    df['Classification_Single'] = np.select(conditions, choices, default='Very Rural')
    df['MinDistance'] = min_distances

    return df


def classify_k_nearest(df, distances, k=5):
    """Classify using average of k nearest distances."""
    distances_copy = distances.copy()
    np.fill_diagonal(distances_copy, np.inf)

    # Get k smallest distances for each hospital
    sorted_distances = np.sort(distances_copy, axis=1)[:, :k]
    mean_distances = np.mean(sorted_distances, axis=1)

    # Classification thresholds (in meters)
    conditions = [
        mean_distances <= 3000,
        mean_distances <= 12000,
        mean_distances <= 20000,
        mean_distances <= 30000,
    ]
    choices = ['Very Urban', 'Urban', 'Suburban', 'Rural']

    df = df.copy()
    df['Classification_KNearest'] = np.select(conditions, choices, default='Very Rural')
    df['MeanDistance'] = mean_distances

    return df


def generate_us_overview():
    """Generate US overview map showing all hospitals."""
    print("Generating US overview map...")

    df = load_hospitals()

    # Sample for performance
    df_sample = df.sample(n=min(3000, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Filter to continental US
    df_plot = df_sample[
        (df_sample['Longitude'] > -130) & (df_sample['Longitude'] < -65) &
        (df_sample['Latitude'] > 24) & (df_sample['Latitude'] < 50)
    ]

    # Plot all hospitals
    scatter = ax.scatter(
        df_plot['Longitude'], df_plot['Latitude'],
        s=np.clip(df_plot['BEDS'] / 10, 5, 50),
        alpha=0.6,
        c='#1c61b6',
        edgecolor='white',
        linewidth=0.3
    )

    ax.set_xlim(-128, -65)
    ax.set_ylim(24, 50)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Hospital Locations Across the Continental United States', fontsize=14)

    # Add size legend
    for beds, size in [(100, 10), (500, 25), (1000, 50)]:
        ax.scatter([], [], s=size, c='#1c61b6', alpha=0.6,
                   label=f'{beds} beds', edgecolor='white')
    ax.legend(title='Hospital Size', loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/us-hospitals-overview.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved us-hospitals-overview.png")


def generate_distance_matrix_example():
    """Generate a visual example of the distance matrix concept."""
    print("Generating distance matrix example...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Simple 4-hospital example
    ax = axes[0]

    # Example hospitals
    hospitals = {
        'A': (0, 0),
        'B': (2, 0),
        'C': (1, 3),
        'D': (5, 4)
    }

    # Plot hospitals
    for name, (x, y) in hospitals.items():
        ax.scatter(x, y, s=200, c='#1c61b6', zorder=5, edgecolor='black', linewidth=2)
        ax.annotate(name, (x, y), fontsize=14, ha='center', va='center',
                   color='white', fontweight='bold')

    # Draw distance lines
    from itertools import combinations
    for (n1, c1), (n2, c2) in combinations(hospitals.items(), 2):
        ax.plot([c1[0], c2[0]], [c1[1], c2[1]], 'k--', alpha=0.3)

    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 5)
    ax.set_xlabel('X coordinate (km)', fontsize=11)
    ax.set_ylabel('Y coordinate (km)', fontsize=11)
    ax.set_title('4 Hospitals in 2D Space', fontsize=12)

    # Right: Distance matrix as heatmap
    ax = axes[1]

    # Calculate distances
    names = list(hospitals.keys())
    n = len(names)
    dist_matrix = np.zeros((n, n))
    for i, (n1, c1) in enumerate(hospitals.items()):
        for j, (n2, c2) in enumerate(hospitals.items()):
            dist_matrix[i, j] = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

    im = ax.imshow(dist_matrix, cmap='YlOrRd')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, fontsize=12)
    ax.set_yticklabels(names, fontsize=12)

    # Add text annotations
    for i in range(n):
        for j in range(n):
            text = f'{dist_matrix[i,j]:.1f}'
            color = 'white' if dist_matrix[i,j] > 3 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=11)

    ax.set_title('Pairwise Distance Matrix (km)', fontsize=12)
    plt.colorbar(im, ax=ax, label='Distance (km)')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/distance-matrix-example.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved distance-matrix-example.png")


def generate_single_vs_knearest():
    """Generate comparison of single distance vs k-nearest classification."""
    print("Generating single vs k-nearest comparison...")

    df = load_hospitals()

    # Focus on a region (e.g., New York area)
    df_region = df[
        (df['Longitude'] > -74.5) & (df['Longitude'] < -73.5) &
        (df['Latitude'] > 40.4) & (df['Latitude'] < 41.2)
    ].copy()

    if len(df_region) < 10:
        print("  Not enough hospitals in region, using sample")
        df_region = df.sample(n=200, random_state=42)

    distances, df_region = calculate_distance_matrix(df_region)
    df_region = classify_single_distance(df_region, distances)
    df_region = classify_k_nearest(df_region, distances, k=5)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, method, title in zip(axes,
                                  ['Classification_Single', 'Classification_KNearest'],
                                  ['Single Minimum Distance', 'K-Nearest Average (k=5)']):

        for category in ['Very Urban', 'Urban', 'Suburban', 'Rural', 'Very Rural']:
            mask = df_region[method] == category
            if mask.sum() > 0:
                ax.scatter(
                    df_region.loc[mask, 'Longitude'],
                    df_region.loc[mask, 'Latitude'],
                    s=np.clip(df_region.loc[mask, 'BEDS'] / 5, 20, 100),
                    c=URBANICITY_COLORS[category],
                    label=category,
                    alpha=0.7,
                    edgecolor='black',
                    linewidth=0.5
                )

        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(title='Classification', loc='upper left', fontsize=9)

    plt.suptitle('New York Region: Classification Method Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/single-vs-knearest.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved single-vs-knearest.png")


def generate_regional_maps():
    """Generate regional comparison maps."""
    print("Generating regional maps...")

    df = load_hospitals()

    # Define regions
    regions = [
        {
            'name': 'New York Metro',
            'lon_range': (-74.5, -73.5),
            'lat_range': (40.4, 41.2)
        },
        {
            'name': 'Rural Montana',
            'lon_range': (-115, -104),
            'lat_range': (44, 49)
        },
        {
            'name': 'Los Angeles',
            'lon_range': (-118.8, -117.5),
            'lat_range': (33.6, 34.4)
        }
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, region in zip(axes, regions):
        df_region = df[
            (df['Longitude'] > region['lon_range'][0]) &
            (df['Longitude'] < region['lon_range'][1]) &
            (df['Latitude'] > region['lat_range'][0]) &
            (df['Latitude'] < region['lat_range'][1])
        ].copy()

        if len(df_region) < 5:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(region['name'])
            continue

        distances, df_region = calculate_distance_matrix(df_region)
        df_region = classify_k_nearest(df_region, distances, k=5)

        for category in ['Very Urban', 'Urban', 'Suburban', 'Rural', 'Very Rural']:
            mask = df_region['Classification_KNearest'] == category
            if mask.sum() > 0:
                ax.scatter(
                    df_region.loc[mask, 'Longitude'],
                    df_region.loc[mask, 'Latitude'],
                    s=np.clip(df_region.loc[mask, 'BEDS'] / 5, 15, 80),
                    c=URBANICITY_COLORS[category],
                    label=category,
                    alpha=0.7,
                    edgecolor='black',
                    linewidth=0.5
                )

        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)
        ax.set_title(region['name'], fontsize=12)

    # Add common legend
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=URBANICITY_COLORS[cat],
                          markersize=10, markeredgecolor='black')
               for cat in ['Very Urban', 'Urban', 'Suburban', 'Rural', 'Very Rural']]
    fig.legend(handles, ['Very Urban', 'Urban', 'Suburban', 'Rural', 'Very Rural'],
              loc='lower center', ncol=5, fontsize=10, bbox_to_anchor=(0.5, -0.05))

    plt.suptitle('Urbanicity Classification Across Different Regions (k=5)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/regional-comparison.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved regional-comparison.png")


def generate_k_sensitivity():
    """Generate plot showing how classification changes with different k values."""
    print("Generating k sensitivity analysis...")

    df = load_hospitals()

    # Sample for performance
    df_sample = df.sample(n=min(1000, len(df)), random_state=42)
    distances, df_sample = calculate_distance_matrix(df_sample)

    k_values = [1, 3, 5, 10, 15]
    results = []

    for k in k_values:
        distances_copy = distances.copy()
        np.fill_diagonal(distances_copy, np.inf)

        if k > distances.shape[0] - 1:
            continue

        sorted_distances = np.sort(distances_copy, axis=1)[:, :k]
        mean_distances = np.mean(sorted_distances, axis=1)

        # Count classifications
        counts = {
            'Very Urban': np.sum(mean_distances <= 3000),
            'Urban': np.sum((mean_distances > 3000) & (mean_distances <= 12000)),
            'Suburban': np.sum((mean_distances > 12000) & (mean_distances <= 20000)),
            'Rural': np.sum((mean_distances > 20000) & (mean_distances <= 30000)),
            'Very Rural': np.sum(mean_distances > 30000)
        }

        for category, count in counts.items():
            results.append({'k': k, 'Category': category, 'Count': count})

    results_df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Very Urban', 'Urban', 'Suburban', 'Rural', 'Very Rural']
    x = np.arange(len(k_values))
    width = 0.15

    for i, category in enumerate(categories):
        cat_data = results_df[results_df['Category'] == category]
        values = [cat_data[cat_data['k'] == k]['Count'].values[0]
                  if len(cat_data[cat_data['k'] == k]) > 0 else 0
                  for k in k_values]
        ax.bar(x + i * width, values, width, label=category,
               color=URBANICITY_COLORS[category], edgecolor='black')

    ax.set_xlabel('Number of Neighbors (k)', fontsize=12)
    ax.set_ylabel('Number of Hospitals', fontsize=12)
    ax.set_title('How k Affects Classification Distribution', fontsize=14)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.legend(title='Classification', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/k-sensitivity.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved k-sensitivity.png")


def generate_classification_legend():
    """Generate a legend/key for the classification scheme."""
    print("Generating classification legend...")

    fig, ax = plt.subplots(figsize=(10, 4))

    categories = [
        ('Very Urban', '< 3 km', 'Dense urban core (Manhattan, downtown Chicago)'),
        ('Urban', '3-12 km', 'City neighborhoods (Brooklyn, suburbs of major cities)'),
        ('Suburban', '12-20 km', 'Suburban areas (outer suburbs, small cities)'),
        ('Rural', '20-30 km', 'Rural areas (small towns, farming regions)'),
        ('Very Rural', '> 30 km', 'Remote areas (frontier regions, wilderness)')
    ]

    y_positions = np.arange(len(categories))[::-1]

    for i, (category, distance, description) in enumerate(categories):
        y = y_positions[i]

        # Draw colored circle
        ax.scatter([0.5], [y], s=500, c=URBANICITY_COLORS[category],
                   edgecolor='black', linewidth=2)

        # Add text
        ax.text(1.2, y, category, fontsize=14, fontweight='bold', va='center')
        ax.text(3, y, f'Mean distance: {distance}', fontsize=12, va='center', style='italic')
        ax.text(5.5, y, description, fontsize=11, va='center')

    ax.set_xlim(0, 12)
    ax.set_ylim(-0.5, len(categories) - 0.5)
    ax.axis('off')
    ax.set_title('Urbanicity Classification Scheme', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/classification-legend.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved classification-legend.png")


def main():
    print("=" * 50)
    print("Generating images for Urbanicity post")
    print("=" * 50)

    generate_us_overview()
    generate_distance_matrix_example()
    generate_single_vs_knearest()
    generate_regional_maps()
    generate_k_sensitivity()
    generate_classification_legend()

    print("\n" + "=" * 50)
    print("All images generated successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
