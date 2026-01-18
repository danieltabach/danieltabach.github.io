---
layout: single
title: "Density Estimation and Mixture Models From Scratch"
date: 2025-01-17
categories: [tutorials]
tags: [machine-learning, density-estimation, kde, gmm, em-algorithm, python, numpy]
author_profile: true
toc: true
toc_label: "Contents"
toc_sticky: true
---

*How do you estimate the shape of data when you don't know what distribution it came from?*

---

## Introduction

Sometimes you have data but no idea what distribution generated it. Maybe it's unimodal, maybe bimodal, maybe something weird. Histograms give you a rough picture, but they're choppy and dependent on bin choices. What if you want a smooth, continuous estimate of the underlying density?

In this post, we build two approaches from scratch:

1. **Kernel Density Estimation (KDE)**: Place a smooth "bump" (kernel) at each data point and sum them up
2. **Gaussian Mixture Models (GMM)**: Assume the data comes from multiple Gaussian distributions and learn their parameters

We'll also implement the **Expectation-Maximization (EM) algorithm**, the standard method for fitting GMMs when you don't know which points belong to which component.

The applications: analyzing brain structure differences across political orientations (surprisingly interesting data), and classifying handwritten digits using GMM as a generative classifier.

**Just want the code?** Skip to the [Appendix](#appendix-complete-implementations) for copy-paste ready implementations.

---

## Part 1: Histograms vs KDE

### The Histogram Problem

Histograms are the most intuitive density estimate. Count how many points fall in each bin, normalize, done. But they have issues:

1. **Bin edges are arbitrary**. Shift your bins slightly and the shape changes.
2. **Not smooth**. Histograms are step functions. Real densities are usually smooth.
3. **Curse of dimensionality**. In 2D you need bins squared, in 3D bins cubed. It gets sparse fast.

### How KDE Works

**Plain English:** Instead of counting points in bins, we place a smooth "bump" centered at each data point. The density estimate at any location is the sum of all these bumps.

**The formula:**

$$\hat{f}(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)$$

Where:
- $n$ is the number of data points
- $h$ is the bandwidth (controls how wide each bump is)
- $K$ is the kernel function (usually Gaussian)
- $x_i$ are your data points

The Gaussian kernel is:

$$K(u) = \frac{1}{\sqrt{2\pi}} e^{-\frac{u^2}{2}}$$

<details>
<summary><strong>See it with a tiny example</strong></summary>
<p>Say we have 3 data points: x = [2, 3, 5] and bandwidth h = 1.</p>
<p>To estimate density at x = 3:</p>
<ul>
<li>Distance from point 1: (3-2)/1 = 1 → K(1) ≈ 0.24</li>
<li>Distance from point 2: (3-3)/1 = 0 → K(0) ≈ 0.40</li>
<li>Distance from point 3: (3-5)/1 = -2 → K(-2) ≈ 0.05</li>
</ul>
<p>Sum: (0.24 + 0.40 + 0.05) / (3 × 1) ≈ 0.23</p>
<p>The density is highest near the data points and smoothly decays away from them.</p>
</details>

### Implementation

```python
import numpy as np
from scipy.stats import gaussian_kde

def kde_1d(data, bandwidth_factor=1.0):
    """
    Compute 1D kernel density estimate.

    Parameters:
    - data: 1D array of observations
    - bandwidth_factor: multiplier for Scott's rule bandwidth

    Returns:
    - kde: scipy gaussian_kde object
    """
    # scipy's gaussian_kde uses Scott's rule by default
    # h = n^(-1/5) * std(data)
    kde = gaussian_kde(data)

    # Adjust bandwidth if needed
    # bw_method scales the default bandwidth
    kde = gaussian_kde(data, bw_method=bandwidth_factor * kde.factor)

    return kde

def kde_2d(x, y, bandwidth_factor=1.0):
    """
    Compute 2D kernel density estimate.

    Parameters:
    - x, y: 1D arrays of coordinates
    - bandwidth_factor: multiplier for default bandwidth

    Returns:
    - kde: scipy gaussian_kde object for 2D data
    """
    # Stack into 2D array where each column is a point
    data = np.vstack([x, y])

    kde = gaussian_kde(data)
    kde = gaussian_kde(data, bw_method=bandwidth_factor * kde.factor)

    return kde
```

**What happens if you change bandwidth?** Too small, and you get a spiky estimate that overfits to individual points. Too large, and you oversmooth, missing real structure like multiple modes. Scott's rule ($h = n^{-1/5} \cdot \sigma$) is a reasonable default, but you often need to tune it.

### Choosing Bandwidth

There's no perfect answer. Common approaches:

| Method | Formula | When to Use |
|--------|---------|-------------|
| Scott's Rule | $h = n^{-1/5} \cdot \sigma$ | Default, assumes roughly Gaussian data |
| Silverman's Rule | $h = 0.9 \cdot \min(\sigma, IQR/1.34) \cdot n^{-1/5}$ | More robust to outliers |
| Cross-validation | Minimize integrated squared error | When you need optimal bandwidth |

In practice, I start with Scott's rule and adjust by eye. If the KDE looks too smooth (missing obvious peaks), decrease the bandwidth. If it's too spiky, increase it.

![Bandwidth comparison](/assets/images/posts/density-estimation/bandwidth-comparison.png)
*Left: Small bandwidth (h=0.3) shows spiky overfit. Middle: Scott's rule bandwidth. Right: Large bandwidth (h=1.0) oversmooths and misses the second mode.*

### Histograms vs KDE: Trade-offs

| Aspect | Histogram | KDE |
|--------|-----------|-----|
| Interpretability | Easy (counts per bin) | Harder (smooth density) |
| Computation | Fast | Slower (scales with n) |
| Smoothness | Discontinuous | Continuous |
| Sampling | Can't easily generate new points | Can sample from the density |
| High dimensions | Breaks down | Also struggles, but degrades more gracefully |

**When to use histograms:** Quick exploration, discrete data, when you need exact counts.

**When to use KDE:** Estimating continuous densities, generating synthetic data, when smoothness matters.

---

## Part 2: Gaussian Mixture Models

### The Problem with Single Gaussians

Sometimes your data has multiple clusters or modes. A single Gaussian can't capture this. If you fit one Gaussian to bimodal data, you get a mean in the middle where no data actually lives.

**Plain English:** A GMM says "your data comes from K different Gaussian distributions, mixed together." Each data point was generated by one of these Gaussians, but we don't know which one.

### The Model

A GMM with $K$ components has:

$$p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$$

Where:
- $\pi_k$ is the mixing weight (probability of choosing component $k$), with $\sum_k \pi_k = 1$
- $\mu_k$ is the mean of component $k$
- $\Sigma_k$ is the covariance matrix of component $k$
- $\mathcal{N}(x | \mu, \Sigma)$ is the multivariate Gaussian density

<details>
<summary><strong>See it with a tiny example</strong></summary>
<p>Say we have K=2 components in 1D:</p>
<table>
<tr><th>Component</th><th>Weight π</th><th>Mean μ</th><th>Variance σ²</th></tr>
<tr><td>1</td><td>0.6</td><td>0</td><td>1</td></tr>
<tr><td>2</td><td>0.4</td><td>5</td><td>2</td></tr>
</table>
<p>The density at x = 2:</p>
<ul>
<li>Component 1: 0.6 × N(2 | 0, 1) = 0.6 × 0.054 = 0.032</li>
<li>Component 2: 0.4 × N(2 | 5, 2) = 0.4 × 0.065 = 0.026</li>
<li>Total: 0.032 + 0.026 = 0.058</li>
</ul>
<p>Point x=2 is somewhat likely under both components, but slightly more likely under component 1.</p>
</details>

### Why MLE Doesn't Work Directly

For a single Gaussian, maximum likelihood estimation is straightforward. You take derivatives, set to zero, and get the sample mean and covariance.

For GMMs, the likelihood is:

$$L(\theta) = \prod_{i=1}^{n} \sum_{k=1}^{K} \pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)$$

Taking the log:

$$\log L(\theta) = \sum_{i=1}^{n} \log \sum_{k=1}^{K} \pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)$$

The problem: that **log of a sum** doesn't simplify nicely. We can't separate terms and solve for each parameter independently. The sum is inside the log.

The solution: **Expectation-Maximization (EM)**.

---

## Part 3: The EM Algorithm

### The Core Idea

EM alternates between two steps:

1. **E-step (Expectation):** Given current parameters, compute the probability that each point belongs to each component. These are called "responsibilities."

2. **M-step (Maximization):** Given responsibilities, update the parameters (means, covariances, weights) to maximize the expected log-likelihood.

Repeat until convergence.

### The Responsibility Formula

Using Bayes' rule, the responsibility $\tau_k^i$ (probability that point $i$ belongs to component $k$) is:

$$\tau_k^i = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}$$

**Plain English:** The numerator is "how likely is this point under component $k$, weighted by how common component $k$ is." The denominator normalizes so responsibilities sum to 1 across components.

<details>
<summary><strong>See it with a tiny example</strong></summary>
<p>Using the same 2-component GMM from before, let's compute responsibilities for x = 2:</p>
<table>
<tr><th>Component</th><th>π × N(x|μ,σ²)</th><th>Responsibility τ</th></tr>
<tr><td>1</td><td>0.6 × 0.054 = 0.032</td><td>0.032 / 0.058 = 0.55</td></tr>
<tr><td>2</td><td>0.4 × 0.065 = 0.026</td><td>0.026 / 0.058 = 0.45</td></tr>
</table>
<p>Point x=2 has 55% responsibility to component 1 and 45% to component 2. It's genuinely ambiguous which component generated it.</p>
</details>

### The M-Step Updates

Given responsibilities, the parameter updates are:

**Effective count for component $k$:**
$$N_k = \sum_{i=1}^{n} \tau_k^i$$

**Updated weights:**
$$\pi_k = \frac{N_k}{n}$$

**Updated means:**
$$\mu_k = \frac{1}{N_k} \sum_{i=1}^{n} \tau_k^i x_i$$

**Updated covariances:**
$$\Sigma_k = \frac{1}{N_k} \sum_{i=1}^{n} \tau_k^i (x_i - \mu_k)(x_i - \mu_k)^T$$

These are just weighted versions of the standard mean and covariance formulas, where each point is weighted by its responsibility to that component.

### Implementation

```python
import numpy as np
from scipy.stats import multivariate_normal

def em_gmm(X, K, max_iters=100, tol=1e-6, seed=None):
    """
    Fit a Gaussian Mixture Model using EM algorithm.

    Parameters:
    - X: data matrix (n_samples, n_features)
    - K: number of components
    - max_iters: maximum iterations
    - tol: convergence tolerance for log-likelihood
    - seed: random seed for initialization

    Returns:
    - pi: mixing weights (K,)
    - mu: means (K, n_features)
    - Sigma: covariances (K, n_features, n_features)
    - responsibilities: final responsibilities (n_samples, K)
    - log_likelihoods: history of log-likelihood values
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples, n_features = X.shape

    # Initialize parameters
    # Means: random Gaussian
    mu = np.random.randn(K, n_features)

    # Covariances: random positive definite matrices
    # Sigma = S @ S.T + I ensures positive definiteness
    Sigma = []
    for k in range(K):
        S = np.random.randn(n_features, n_features)
        Sigma.append(S @ S.T + np.eye(n_features))

    # Weights: uniform
    pi = np.ones(K) / K

    log_likelihoods = []

    for iteration in range(max_iters):
        # E-Step: Compute responsibilities
        # gamma[i, k] = P(point i belongs to component k)
        gamma = np.zeros((n_samples, K))

        for k in range(K):
            # Numerator: pi_k * N(x | mu_k, Sigma_k)
            gamma[:, k] = pi[k] * multivariate_normal.pdf(
                X, mean=mu[k], cov=Sigma[k]
            )

        # Normalize: divide by sum across components
        gamma /= gamma.sum(axis=1, keepdims=True)

        # M-Step: Update parameters
        # Effective count per component
        Nk = gamma.sum(axis=0)

        # Update weights
        pi = Nk / n_samples

        # Update means and covariances
        for k in range(K):
            # Weighted mean
            mu[k] = np.dot(gamma[:, k], X) / Nk[k]

            # Weighted covariance
            centered = X - mu[k]
            Sigma[k] = np.dot((gamma[:, k, None] * centered).T, centered) / Nk[k]

        # Compute log-likelihood
        ll = 0
        for i in range(n_samples):
            point_likelihood = sum(
                pi[k] * multivariate_normal.pdf(X[i], mean=mu[k], cov=Sigma[k])
                for k in range(K)
            )
            ll += np.log(point_likelihood)

        log_likelihoods.append(ll)

        # Check convergence
        if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break

    return pi, mu, Sigma, gamma, log_likelihoods
```

**What happens if you change K?** Too few components and you underfit, missing real structure. Too many and you overfit, potentially assigning separate components to noise. Use BIC or AIC for model selection (lower is better):

$$BIC = -2 \log L + p \log n$$
$$AIC = -2 \log L + 2p$$

Where $p$ is the number of parameters. BIC penalizes complexity more heavily.

### Handling Edge Cases

**Singular covariance matrices:** If a component collapses to a single point, the covariance becomes singular. Add a small regularization term: $\Sigma_k + \epsilon I$ where $\epsilon \approx 10^{-6}$.

**Empty components:** If a component gets zero responsibility, reinitialize it randomly or remove it.

**Local minima:** EM finds local optima. Run multiple times with different initializations and keep the best (highest log-likelihood).

---

## Application: Brain Structure and Political Views

### The Dataset

We have data on 90 subjects with three variables:
- **amygdala**: volume of the amygdala (brain region associated with emotion)
- **acc**: volume of the anterior cingulate cortex (associated with decision-making)
- **orientation**: political orientation on a scale from 2 (conservative) to 5 (liberal)

The scientific question: Is there a relationship between brain structure and political orientation?

### 1D Analysis

First, let's look at each variable separately using histograms and KDE:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('n90pol.csv')

# Compute number of bins using Sturges' rule
n = len(data)
n_bins = int(np.floor(np.log2(n) + 1))

# Plot histograms
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(data['amygdala'], bins=n_bins, ax=axes[0])
axes[0].set_title('Histogram of Amygdala Volume')

sns.histplot(data['acc'], bins=n_bins, ax=axes[1])
axes[1].set_title('Histogram of ACC Volume')

plt.show()

# Plot KDE
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.kdeplot(data['amygdala'], bw_adjust=0.5, ax=axes[0])
axes[0].set_title('KDE of Amygdala Volume')

sns.kdeplot(data['acc'], bw_adjust=0.5, ax=axes[1])
axes[1].set_title('KDE of ACC Volume')

plt.show()
```

The KDE reveals potential multimodality in the amygdala distribution that the histogram hints at but doesn't show clearly.

![1D Histogram and KDE](/assets/images/posts/density-estimation/amygdala-acc-1d.png)
*Top row: Histograms using Sturges' rule for bin count. Bottom row: KDE with bandwidth factor 0.5. The KDE reveals a potential secondary mode in the amygdala distribution.*

### 2D Analysis

Looking at both variables together:

```python
from scipy.stats import gaussian_kde

X = data['amygdala'].values
Y = data['acc'].values

# Compute 2D KDE
kde = gaussian_kde(np.vstack([X, Y]), bw_method=0.3)

# Create grid for evaluation
x_grid, y_grid = np.mgrid[X.min():X.max():100j, Y.min():Y.max():100j]
positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
z = kde(positions).reshape(x_grid.shape)

# Plot
plt.figure(figsize=(8, 6))
plt.contourf(x_grid, y_grid, z, levels=20, cmap='plasma')
plt.colorbar(label='Density')
plt.xlabel('Amygdala Volume')
plt.ylabel('ACC Volume')
plt.title('2D KDE of Brain Structure')
plt.show()
```

![2D KDE Contour](/assets/images/posts/density-estimation/brain-2d-kde.png)
*2D KDE of amygdala vs ACC volume. The contours suggest a primary mode with possible secondary structure. Red ellipses mark potential modes; green ellipses mark outliers.*

![3D KDE Surface](/assets/images/posts/density-estimation/brain-3d-kde.png)
*3D surface plot of the same density. The peaks and valleys are easier to see in this view.*

### Testing for Independence

Are amygdala and acc volumes independent?

```python
from scipy.stats import pearsonr

corr, p_value = pearsonr(data['amygdala'], data['acc'])
print(f'Pearson correlation: {corr:.3f}')
print(f'p-value: {p_value:.3f}')
```

Results:
- Correlation: -0.13
- p-value: 0.23

The correlation is weak and not statistically significant. We can't reject the null hypothesis that the variables are independent.

### Conditional Distributions by Political Orientation

The more interesting question: do the distributions differ by political orientation?

```python
# Plot conditional KDEs for each orientation
orientations = [2, 3, 4, 5]

plt.figure(figsize=(12, 6))
for c in orientations:
    subset = data[data['orientation'] == c]['amygdala']
    sns.kdeplot(subset, label=f'Orientation {c}', bw_adjust=0.6)

plt.xlabel('Amygdala Volume')
plt.ylabel('Density')
plt.title('Amygdala Volume by Political Orientation')
plt.legend()
plt.show()
```

![Conditional KDE by Orientation](/assets/images/posts/density-estimation/conditional-kde-orientation.png)
*KDE of amygdala volume conditioned on political orientation. Orientation 2 (conservative) shows a distinct rightward shift compared to 4 and 5 (liberal).*

| Orientation | Amygdala Mean | ACC Mean | Observations |
|-------------|---------------|----------|--------------|
| 2 (Conservative) | 0.019 | -0.015 | Larger amygdala, smaller ACC |
| 3 (Lean Conservative) | 0.001 | 0.002 | Near average |
| 4 (Lean Liberal) | -0.005 | 0.001 | Smaller amygdala |
| 5 (Liberal) | -0.006 | 0.008 | Smaller amygdala, larger ACC |

The conditional 2D KDE plots show distinct patterns:
- **Orientation 3** (neutral) has the widest spread in both dimensions
- **Orientation 2** and **4** (leaning groups) show similar outlier patterns
- **Orientation 5** (most liberal) has the narrowest distribution

![Joint KDE by Orientation](/assets/images/posts/density-estimation/joint-kde-by-orientation.png)
*2D KDE of (amygdala, acc) for each political orientation. Orientation 3 (neutral) shows the widest spread, while more partisan orientations have tighter clusters.*

### Model Selection: How Many Components?

Using sklearn's GaussianMixture to test whether the joint distribution is unimodal or multimodal:

```python
from sklearn.mixture import GaussianMixture

X_joint = data[['amygdala', 'acc']].values

# Test different numbers of components
n_components_range = range(1, 7)
bics = []
aics = []

for n in n_components_range:
    gmm = GaussianMixture(n_components=n, random_state=0)
    gmm.fit(X_joint)
    bics.append(gmm.bic(X_joint))
    aics.append(gmm.aic(X_joint))

# Find optimal
best_bic = n_components_range[np.argmin(bics)]
best_aic = n_components_range[np.argmin(aics)]
print(f'Optimal components (BIC): {best_bic}')
print(f'Optimal components (AIC): {best_aic}')
```

Results:
- BIC suggests 1 component
- AIC suggests 3 components

![BIC and AIC Model Selection](/assets/images/posts/density-estimation/bic-aic-components.png)
*BIC and AIC for different numbers of GMM components. BIC (which penalizes complexity more) favors 1 component, while AIC suggests 3.*

BIC penalizes complexity more heavily and is generally preferred for model selection. Despite what the KDE visually suggests, the statistical evidence doesn't strongly support multiple components. This is a good reminder: visual patterns in small samples can be noise.

---

## Application: Digit Classification with GMM

### The Problem

We have images of handwritten digits (2 and 6 from MNIST). Can we classify them using GMM as a generative model?

The approach:
1. Reduce dimensionality with PCA (784 pixels → 4 components)
2. Fit a 2-component GMM
3. Assign each digit to the component with higher responsibility
4. Map components to digit labels

### Loading and Preprocessing

```python
from scipy.io import loadmat
from sklearn.decomposition import PCA

# Load data
data_mat = loadmat('data.mat')
labels_mat = loadmat('label.mat')

X = data_mat['data'].T  # Shape: (1990, 784)
y = labels_mat['trueLabel'].flatten()  # Labels: 2 or 6

print(f'Data shape: {X.shape}')
print(f'Unique labels: {np.unique(y)}')

# Apply PCA to reduce from 784D to 4D
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X)

print(f'PCA shape: {X_pca.shape}')
```

### Fitting the GMM

```python
# Run EM algorithm
pi, mu, Sigma, gamma, log_likelihoods = em_gmm(X_pca, K=2, seed=1)

# Plot convergence
plt.figure(figsize=(8, 5))
plt.plot(log_likelihoods)
plt.xlabel('Iteration')
plt.ylabel('Log-Likelihood')
plt.title('EM Algorithm Convergence')
plt.show()

print(f'Final weights: {pi}')
print(f'Converged in {len(log_likelihoods)} iterations')
```

Results:
- Weight 1: 0.51 (roughly half the data)
- Weight 2: 0.49
- Converged in ~18 iterations

![EM Convergence](/assets/images/posts/density-estimation/em-convergence.png)
*Log-likelihood increases monotonically and plateaus around iteration 15. The algorithm converges quickly for this well-separated data.*

### Visualizing the Learned Means

The mean of each component, when mapped back to image space, should look like an "average" digit:

```python
# Map means back to original space
mu_original = pca.inverse_transform(mu)

# Display as images
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for i in range(2):
    axes[i].imshow(mu_original[i].reshape(28, 28), cmap='gray')
    axes[i].set_title(f'Mean Image of Cluster {i+1}')
    axes[i].axis('off')
plt.show()
```

![Learned Mean Images](/assets/images/posts/density-estimation/gmm-mean-images.png)
*Mean images of the two GMM components, mapped back to 28x28 pixel space. Left: clearly resembles a 2. Right: resembles a 6 (slightly tilted due to PCA reconstruction loss).*

The learned means clearly resemble a 2 and a 6. The GMM has discovered the digit structure without being told the labels.

### Covariance Structure

The 4x4 covariance matrices show how the principal components co-vary within each cluster:

```python
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for i in range(2):
    sns.heatmap(Sigma[i], annot=True, fmt='.2f', ax=axes[i], cmap='RdBu_r')
    axes[i].set_title(f'Covariance Matrix of Cluster {i+1}')
plt.tight_layout()
plt.show()
```

![Covariance Heatmaps](/assets/images/posts/density-estimation/covariance-heatmaps.png)
*4x4 covariance matrices for each component in PCA space. Different patterns reflect how digit 2 and digit 6 vary along the principal components.*

The covariance matrices differ between clusters, capturing that 2s and 6s vary differently across the principal components.

### Classification Results

```python
from sklearn.metrics import classification_report, confusion_matrix

# Assign each point to the component with higher responsibility
predicted_clusters = np.argmax(gamma, axis=1)

# Map clusters to digit labels based on majority
# (cluster 0 might correspond to digit 6, cluster 1 to digit 2, or vice versa)
cluster_0_labels = y[predicted_clusters == 0]
cluster_1_labels = y[predicted_clusters == 1]

# Determine mapping
if np.sum(cluster_0_labels == 2) > np.sum(cluster_0_labels == 6):
    cluster_to_digit = {0: 2, 1: 6}
else:
    cluster_to_digit = {0: 6, 1: 2}

predicted_digits = np.array([cluster_to_digit[c] for c in predicted_clusters])

# Evaluate
print('GMM Classification Results:')
print(confusion_matrix(y, predicted_digits))
print(classification_report(y, predicted_digits, target_names=['Digit 2', 'Digit 6']))
```

| Metric | GMM | K-Means |
|--------|-----|---------|
| Accuracy | 96% | 93% |
| Precision (Digit 2) | 0.99 | 0.93 |
| Recall (Digit 2) | 0.94 | 0.94 |
| Precision (Digit 6) | 0.93 | 0.93 |
| Recall (Digit 6) | 0.99 | 0.92 |

GMM outperforms K-Means across all metrics. Why? GMM models the full covariance structure, allowing elliptical clusters. K-Means assumes spherical clusters with equal variance in all directions.

---

## Summary

### Histograms
- Simple and interpretable
- Sensitive to bin placement
- Discontinuous, can't generate new samples
- Breaks down in high dimensions

### Kernel Density Estimation
- Smooth, continuous density estimate
- Bandwidth is the key parameter (use Scott's rule as starting point)
- Can sample from the estimated distribution
- Computationally expensive for large datasets

### Gaussian Mixture Models
- Model data as a mixture of Gaussian components
- Learn parameters with EM algorithm
- Soft assignments: each point has probability of belonging to each component
- More flexible than K-Means (allows elliptical clusters)

### EM Algorithm
- E-step: compute responsibilities using current parameters
- M-step: update parameters using responsibilities
- Guaranteed to increase log-likelihood each iteration
- Converges to local optimum (run multiple times)

### General Principles

1. **Bandwidth matters in KDE.** Too small = overfitting, too large = oversmoothing. Start with Scott's rule and adjust.

2. **Use BIC/AIC for model selection.** Don't just pick the number of components that looks right visually.

3. **GMM beats K-Means when clusters are elliptical.** If your clusters are truly spherical, K-Means is simpler and faster.

4. **EM finds local optima.** Run multiple times with different initializations.

5. **Visual patterns can be noise.** Statistical tests often reveal that apparent structure isn't significant, especially in small samples.

---

## References

**Kernel Density Estimation**
- Silverman, B. W. (1986). *Density Estimation for Statistics and Data Analysis*. Chapman and Hall.
- Scott, D. W. (1992). *Multivariate Density Estimation: Theory, Practice, and Visualization*. Wiley.

**Gaussian Mixture Models and EM**
- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). ["Maximum Likelihood from Incomplete Data via the EM Algorithm."](https://www.jstor.org/stable/2984875) *Journal of the Royal Statistical Society: Series B*, 39(1), 1-38.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 9. Springer.

**Model Selection**
- Schwarz, G. (1978). "Estimating the Dimension of a Model." *Annals of Statistics*, 6(2), 461-464.
- Akaike, H. (1974). "A New Look at the Statistical Model Identification." *IEEE Transactions on Automatic Control*, 19(6), 716-723.

---

## Appendix: Complete Implementations

### KDE Implementation

```python
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

def kde_1d(data, bandwidth_factor=1.0):
    """1D Kernel Density Estimation."""
    kde = gaussian_kde(data)
    kde = gaussian_kde(data, bw_method=bandwidth_factor * kde.factor)
    return kde

def kde_2d(x, y, bandwidth_factor=1.0):
    """2D Kernel Density Estimation."""
    data = np.vstack([x, y])
    kde = gaussian_kde(data)
    kde = gaussian_kde(data, bw_method=bandwidth_factor * kde.factor)
    return kde

def plot_kde_2d(x, y, bandwidth_factor=0.5, grid_size=100):
    """Plot 2D KDE as contour."""
    kde = kde_2d(x, y, bandwidth_factor)

    x_grid, y_grid = np.mgrid[x.min():x.max():complex(grid_size),
                               y.min():y.max():complex(grid_size)]
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
    z = kde(positions).reshape(x_grid.shape)

    plt.contourf(x_grid, y_grid, z, levels=20, cmap='plasma')
    plt.colorbar(label='Density')
    return kde
```

### GMM with EM Implementation

```python
import numpy as np
from scipy.stats import multivariate_normal

def em_gmm(X, K, max_iters=100, tol=1e-6, seed=None):
    """
    Fit GMM using Expectation-Maximization.

    Parameters:
    - X: data (n_samples, n_features)
    - K: number of components
    - max_iters: maximum iterations
    - tol: convergence tolerance
    - seed: random seed

    Returns:
    - pi: weights (K,)
    - mu: means (K, n_features)
    - Sigma: covariances list of (n_features, n_features)
    - gamma: responsibilities (n_samples, K)
    - log_likelihoods: convergence history
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples, n_features = X.shape

    # Initialize
    mu = np.random.randn(K, n_features)
    Sigma = []
    for k in range(K):
        S = np.random.randn(n_features, n_features)
        Sigma.append(S @ S.T + np.eye(n_features))
    pi = np.ones(K) / K

    log_likelihoods = []

    for iteration in range(max_iters):
        # E-Step
        gamma = np.zeros((n_samples, K))
        for k in range(K):
            gamma[:, k] = pi[k] * multivariate_normal.pdf(X, mean=mu[k], cov=Sigma[k])
        gamma /= gamma.sum(axis=1, keepdims=True)

        # M-Step
        Nk = gamma.sum(axis=0)
        pi = Nk / n_samples

        for k in range(K):
            mu[k] = np.dot(gamma[:, k], X) / Nk[k]
            centered = X - mu[k]
            Sigma[k] = np.dot((gamma[:, k, None] * centered).T, centered) / Nk[k]

        # Log-likelihood
        ll = np.sum(np.log(np.sum([
            pi[k] * multivariate_normal.pdf(X, mean=mu[k], cov=Sigma[k])
            for k in range(K)
        ], axis=0)))
        log_likelihoods.append(ll)

        # Convergence check
        if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break

    return pi, mu, Sigma, gamma, log_likelihoods

def predict_gmm(X, pi, mu, Sigma):
    """Predict cluster assignments from fitted GMM."""
    K = len(pi)
    n_samples = X.shape[0]

    gamma = np.zeros((n_samples, K))
    for k in range(K):
        gamma[:, k] = pi[k] * multivariate_normal.pdf(X, mean=mu[k], cov=Sigma[k])
    gamma /= gamma.sum(axis=1, keepdims=True)

    return np.argmax(gamma, axis=1), gamma
```

### Model Selection

```python
from sklearn.mixture import GaussianMixture

def select_n_components(X, max_components=10):
    """Select optimal number of GMM components using BIC."""
    bics = []
    aics = []

    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, random_state=0)
        gmm.fit(X)
        bics.append(gmm.bic(X))
        aics.append(gmm.aic(X))

    best_bic = np.argmin(bics) + 1
    best_aic = np.argmin(aics) + 1

    return best_bic, best_aic, bics, aics
```
