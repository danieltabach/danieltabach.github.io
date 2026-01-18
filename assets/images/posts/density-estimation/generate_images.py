"""
Generate all images for the Density Estimation blog post.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, multivariate_normal
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

# Paths
DATA_PATH = r"C:\Users\Interview Prep\Desktop\Portfolio Github\CDA Homeworks Mini\Submission Files HW3\Data"
OUTPUT_PATH = r"C:\Users\Interview Prep\Desktop\Portfolio Github\danieltabach.github.io\assets\images\posts\density-estimation"

# ============================================================
# 1. Bandwidth Comparison
# ============================================================
print("Generating bandwidth comparison...")

np.random.seed(42)
# Create bimodal data
data_bw = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(4, 0.8, 60)])

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
bandwidths = [0.3, 1.0, 2.0]
titles = ['Small bandwidth (h=0.3)\nOverfit - too spiky',
          "Scott's rule bandwidth\nBalanced",
          'Large bandwidth (h=2.0)\nOversmooth - misses second mode']

x_range = np.linspace(-4, 8, 200)

for ax, bw, title in zip(axes, bandwidths, titles):
    kde = gaussian_kde(data_bw, bw_method=bw * gaussian_kde(data_bw).factor)
    ax.hist(data_bw, bins=20, density=True, alpha=0.3, color='gray', label='Histogram')
    ax.plot(x_range, kde(x_range), 'b-', linewidth=2, label='KDE')
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/bandwidth-comparison.png", bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved bandwidth-comparison.png")

# ============================================================
# 2. Load n90pol data
# ============================================================
print("Loading n90pol data...")
n90pol = pd.read_csv(f"{DATA_PATH}/n90pol.csv")
print(f"  Loaded {len(n90pol)} samples")

# ============================================================
# 3. 1D Histograms and KDE for amygdala and acc
# ============================================================
print("Generating 1D histograms and KDE...")

n = len(n90pol)
sturges_bins = int(np.floor(np.log2(n) + 1))

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histograms
axes[0, 0].hist(n90pol['amygdala'], bins=sturges_bins, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].set_title('Histogram of Amygdala Volume')
axes[0, 0].set_xlabel('Amygdala Volume')
axes[0, 0].set_ylabel('Frequency')

axes[0, 1].hist(n90pol['acc'], bins=sturges_bins, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 1].set_title('Histogram of ACC Volume')
axes[0, 1].set_xlabel('ACC Volume')
axes[0, 1].set_ylabel('Frequency')

# KDE
x_amy = np.linspace(n90pol['amygdala'].min() - 0.02, n90pol['amygdala'].max() + 0.02, 200)
kde_amy = gaussian_kde(n90pol['amygdala'], bw_method=0.5 * gaussian_kde(n90pol['amygdala']).factor)
axes[1, 0].plot(x_amy, kde_amy(x_amy), 'b-', linewidth=2)
axes[1, 0].fill_between(x_amy, kde_amy(x_amy), alpha=0.3)
axes[1, 0].set_title('KDE of Amygdala Volume (bw_adjust=0.5)')
axes[1, 0].set_xlabel('Amygdala Volume')
axes[1, 0].set_ylabel('Density')

x_acc = np.linspace(n90pol['acc'].min() - 0.02, n90pol['acc'].max() + 0.02, 200)
kde_acc = gaussian_kde(n90pol['acc'], bw_method=0.5 * gaussian_kde(n90pol['acc']).factor)
axes[1, 1].plot(x_acc, kde_acc(x_acc), 'b-', linewidth=2)
axes[1, 1].fill_between(x_acc, kde_acc(x_acc), alpha=0.3)
axes[1, 1].set_title('KDE of ACC Volume (bw_adjust=0.5)')
axes[1, 1].set_xlabel('ACC Volume')
axes[1, 1].set_ylabel('Density')

plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/amygdala-acc-1d.png", bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved amygdala-acc-1d.png")

# ============================================================
# 4. 2D KDE Contour
# ============================================================
print("Generating 2D KDE contour...")

X = n90pol['amygdala'].values
Y = n90pol['acc'].values

kde_2d = gaussian_kde(np.vstack([X, Y]), bw_method=0.3)

x_min, x_max = X.min() - 0.01, X.max() + 0.01
y_min, y_max = Y.min() - 0.01, Y.max() + 0.01

x_grid, y_grid = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
z = kde_2d(positions).reshape(x_grid.shape)

fig, ax = plt.subplots(figsize=(10, 8))
contour = ax.contourf(x_grid, y_grid, z, levels=20, cmap='plasma')
plt.colorbar(contour, label='Density')
ax.scatter(X, Y, c='white', s=20, alpha=0.5, edgecolors='black', linewidths=0.5)
ax.set_xlabel('Amygdala Volume')
ax.set_ylabel('ACC Volume')
ax.set_title('2D Kernel Density Estimation (KDE) for Amygdala vs. ACC')

# Add ellipses to mark modes
from matplotlib.patches import Ellipse
ellipse1 = Ellipse(xy=(-0.015, -0.012), width=0.055, height=0.02, angle=-30,
                   edgecolor='red', facecolor='none', linewidth=2, label='Potential modes')
ellipse2 = Ellipse(xy=(0.01, 0.005), width=0.06, height=0.02, angle=-30,
                   edgecolor='red', facecolor='none', linewidth=2)
ellipse3 = Ellipse(xy=(-0.06, 0.00), width=0.015, height=0.015, angle=-50,
                   edgecolor='lime', facecolor='none', linewidth=2, label='Outliers')
ellipse4 = Ellipse(xy=(-0.018, 0.05), width=0.015, height=0.015, angle=-50,
                   edgecolor='lime', facecolor='none', linewidth=2)

ax.add_patch(ellipse1)
ax.add_patch(ellipse2)
ax.add_patch(ellipse3)
ax.add_patch(ellipse4)
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/brain-2d-kde.png", bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved brain-2d-kde.png")

# ============================================================
# 5. 3D KDE Surface
# ============================================================
print("Generating 3D KDE surface...")

kde_3d = gaussian_kde(np.vstack([X, Y]), bw_method=0.65 * gaussian_kde(np.vstack([X, Y])).factor)

xs, ys = np.mgrid[x_min:x_max:50j, y_min:y_max:50j]
zs = kde_3d(np.array([xs.ravel(), ys.ravel()])).reshape(xs.shape)

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xs, ys, zs, cmap='hot_r', linewidth=0.5, rstride=1, cstride=1, edgecolor='k', alpha=0.9)
ax.set_xlabel('Amygdala Volume')
ax.set_ylabel('ACC Volume')
ax.set_zlabel('Density')
ax.set_title('3D KDE Density Plot of Amygdala vs. ACC')
ax.view_init(elev=25, azim=45)

plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/brain-3d-kde.png", bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved brain-3d-kde.png")

# ============================================================
# 6. Conditional KDE by Orientation
# ============================================================
print("Generating conditional KDE by orientation...")

orientations = [2, 3, 4, 5]
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Amygdala
for c, color in zip(orientations, colors):
    subset = n90pol[n90pol['orientation'] == c]['amygdala']
    kde = gaussian_kde(subset, bw_method=0.6 * gaussian_kde(subset).factor)
    x_range = np.linspace(subset.min() - 0.02, subset.max() + 0.02, 200)
    axes[0].plot(x_range, kde(x_range), color=color, linewidth=2, label=f'Orientation {c}')
    axes[0].fill_between(x_range, kde(x_range), alpha=0.1, color=color)

axes[0].set_xlabel('Amygdala Volume')
axes[0].set_ylabel('Density')
axes[0].set_title('Conditional KDE: Amygdala Volume by Orientation')
axes[0].legend()

# ACC
for c, color in zip(orientations, colors):
    subset = n90pol[n90pol['orientation'] == c]['acc']
    kde = gaussian_kde(subset, bw_method=0.6 * gaussian_kde(subset).factor)
    x_range = np.linspace(subset.min() - 0.02, subset.max() + 0.02, 200)
    axes[1].plot(x_range, kde(x_range), color=color, linewidth=2, label=f'Orientation {c}')
    axes[1].fill_between(x_range, kde(x_range), alpha=0.1, color=color)

axes[1].set_xlabel('ACC Volume')
axes[1].set_ylabel('Density')
axes[1].set_title('Conditional KDE: ACC Volume by Orientation')
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/conditional-kde-orientation.png", bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved conditional-kde-orientation.png")

# ============================================================
# 7. Joint KDE by Orientation (2x2 grid)
# ============================================================
print("Generating joint KDE by orientation...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, c in enumerate(orientations):
    subset = n90pol[n90pol['orientation'] == c]
    X_sub = subset['amygdala'].values
    Y_sub = subset['acc'].values

    if len(X_sub) > 5:  # Need enough points for KDE
        kde = gaussian_kde(np.vstack([X_sub, Y_sub]))

        x_grid, y_grid = np.mgrid[X_sub.min()-0.02:X_sub.max()+0.02:50j,
                                   Y_sub.min()-0.02:Y_sub.max()+0.02:50j]
        positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
        z = kde(positions).reshape(x_grid.shape)

        axes[idx].contourf(x_grid, y_grid, z, levels=15, cmap='plasma')
        axes[idx].scatter(X_sub, Y_sub, c='white', s=15, alpha=0.6, edgecolors='black', linewidths=0.3)

    axes[idx].set_xlabel('Amygdala Volume')
    axes[idx].set_ylabel('ACC Volume')
    axes[idx].set_title(f'Orientation = {c}')

plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/joint-kde-by-orientation.png", bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved joint-kde-by-orientation.png")

# ============================================================
# 8. BIC/AIC Model Selection
# ============================================================
print("Generating BIC/AIC model selection plot...")

X_joint = n90pol[['amygdala', 'acc']].values
n_components_range = range(1, 7)

bics = []
aics = []

for n in n_components_range:
    gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=0)
    gmm.fit(X_joint)
    bics.append(gmm.bic(X_joint))
    aics.append(gmm.aic(X_joint))

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(list(n_components_range), bics, 'o-', linewidth=2, markersize=8, label='BIC', color='#1f77b4')
ax.plot(list(n_components_range), aics, 's--', linewidth=2, markersize=8, label='AIC', color='#ff7f0e')
ax.axvline(x=np.argmin(bics)+1, color='#1f77b4', linestyle=':', alpha=0.7, label=f'Best BIC (k={np.argmin(bics)+1})')
ax.axvline(x=np.argmin(aics)+1, color='#ff7f0e', linestyle=':', alpha=0.7, label=f'Best AIC (k={np.argmin(aics)+1})')
ax.set_xlabel('Number of Components')
ax.set_ylabel('Criterion Value')
ax.set_title('BIC and AIC for Different Numbers of GMM Components')
ax.legend()
ax.set_xticks(list(n_components_range))

plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/bic-aic-components.png", bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved bic-aic-components.png")

# ============================================================
# 9. Load digit data and run EM
# ============================================================
print("Loading digit data...")

data_mat = loadmat(f"{DATA_PATH}/data.mat")
labels_mat = loadmat(f"{DATA_PATH}/label.mat")

x_digits = data_mat['data'].T  # (1990, 784)
y_digits = labels_mat['trueLabel'].flatten()

print(f"  Loaded {x_digits.shape[0]} digit images")

# Apply PCA
pca = PCA(n_components=4)
x_pca = pca.fit_transform(x_digits)

# ============================================================
# 10. Run EM Algorithm
# ============================================================
print("Running EM algorithm...")

np.random.seed(1)
K = 2
n_samples, n_features = x_pca.shape

# Initialize
mu = np.random.randn(K, n_features)
S1 = np.random.randn(n_features, n_features)
S2 = np.random.randn(n_features, n_features)
Sigma = [S1 @ S1.T + np.eye(n_features), S2 @ S2.T + np.eye(n_features)]
pi = np.array([0.5, 0.5])

max_epochs = 100
log_likelihoods = []

for i in range(max_epochs):
    # E-Step
    gamma = np.zeros((n_samples, K))
    for k in range(K):
        gamma[:, k] = pi[k] * multivariate_normal.pdf(x_pca, mean=mu[k], cov=Sigma[k])
    gamma /= gamma.sum(axis=1, keepdims=True)

    # M-Step
    Nk = gamma.sum(axis=0)
    pi = Nk / n_samples

    for k in range(K):
        mu[k] = np.dot(gamma[:, k], x_pca) / Nk[k]
        centered = x_pca - mu[k]
        Sigma[k] = np.dot((gamma[:, k, None] * centered).T, centered) / Nk[k]

    # Log-likelihood
    ll = np.sum(np.log(np.sum([
        pi[k] * multivariate_normal.pdf(x_pca, mean=mu[k], cov=Sigma[k])
        for k in range(K)
    ], axis=0)))
    log_likelihoods.append(ll)

    if i > 1 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < 1e-6:
        break

print(f"  Converged in {len(log_likelihoods)} iterations")

# ============================================================
# 11. EM Convergence Plot
# ============================================================
print("Generating EM convergence plot...")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(log_likelihoods, 'b-', linewidth=2, marker='o', markersize=4)
ax.set_xlabel('Iteration')
ax.set_ylabel('Log-Likelihood')
ax.set_title('EM Algorithm Convergence')
ax.axhline(y=log_likelihoods[-1], color='r', linestyle='--', alpha=0.7, label=f'Converged: {log_likelihoods[-1]:.1f}')
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/em-convergence.png", bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved em-convergence.png")

# ============================================================
# 12. GMM Mean Images
# ============================================================
print("Generating GMM mean images...")

mu_original = pca.inverse_transform(mu)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for i in range(2):
    img = mu_original[i].reshape(28, 28)
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f'Mean Image of Cluster {i+1}', fontsize=14)
    axes[i].axis('off')

plt.suptitle('Learned GMM Component Means (Mapped Back to Image Space)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/gmm-mean-images.png", bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved gmm-mean-images.png")

# ============================================================
# 13. Covariance Heatmaps
# ============================================================
print("Generating covariance heatmaps...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i in range(2):
    sns.heatmap(Sigma[i], annot=True, fmt='.2f', ax=axes[i], cmap='RdBu_r', center=0,
                xticklabels=['PC1', 'PC2', 'PC3', 'PC4'],
                yticklabels=['PC1', 'PC2', 'PC3', 'PC4'])
    axes[i].set_title(f'Covariance Matrix of Cluster {i+1}', fontsize=12)

plt.suptitle('4x4 Covariance Matrices in PCA Space', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PATH}/covariance-heatmaps.png", bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved covariance-heatmaps.png")

print("\nAll images generated successfully!")
