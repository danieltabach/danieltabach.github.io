"""
Generate images for the Classification From Scratch blog post.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

# Data path
DATA_PATH = r"C:\Users\Interview Prep\Desktop\Portfolio Github\CDA Homeworks Mini\Submission Files HW4\data\marriage.csv"

def load_data():
    """Load the marriage dataset."""
    data = pd.read_csv(DATA_PATH)

    # Get feature columns (assuming columns are named Feature1, Feature2, etc.)
    feature_cols = [col for col in data.columns if col != 'Label']
    X = data[feature_cols].values
    y = data['Label'].values

    return X, y

def generate_sigmoid_plot():
    """Generate plot showing the sigmoid function."""
    print("Generating sigmoid function plot...")

    fig, ax = plt.subplots(figsize=(8, 5))

    z = np.linspace(-6, 6, 200)
    sigmoid = 1 / (1 + np.exp(-z))

    ax.plot(z, sigmoid, 'b-', linewidth=2, label=r'$\sigma(z) = \frac{1}{1+e^{-z}}$')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('z', fontsize=12)
    ax.set_ylabel(r'$\sigma(z)$', fontsize=12)
    ax.set_title('The Sigmoid Function', fontsize=14)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('sigmoid-function.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved sigmoid-function.png")

def generate_gradient_descent_convergence():
    """Generate plot showing gradient descent convergence."""
    print("Generating gradient descent convergence plot...")

    X, y = load_data()

    # Add bias term
    X_bias = np.hstack([np.ones((X.shape[0], 1)), X])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_bias, y, test_size=0.2, random_state=42
    )

    # Standardize features
    mean = X_train[:, 1:].mean(axis=0)
    std = X_train[:, 1:].std(axis=0) + 1e-10
    X_train[:, 1:] = (X_train[:, 1:] - mean) / std

    # Run gradient descent and track log-likelihood
    m, d = X_train.shape
    theta = np.zeros(d)
    learning_rate = 0.1
    max_iters = 100

    log_likelihoods = []

    for iteration in range(max_iters):
        z = X_train @ theta
        z = np.clip(z, -500, 500)  # Prevent overflow
        predictions = 1 / (1 + np.exp(-z))

        # Log-likelihood
        ll = np.sum(y_train * np.log(predictions + 1e-10) +
                   (1 - y_train) * np.log(1 - predictions + 1e-10))
        log_likelihoods.append(ll)

        # Gradient
        gradient = X_train.T @ (y_train - predictions)

        # Update
        theta = theta + learning_rate * gradient

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(log_likelihoods) + 1), log_likelihoods, 'b-', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Log-Likelihood', fontsize=12)
    ax.set_title('Gradient Descent Convergence for Logistic Regression', fontsize=14)

    plt.tight_layout()
    plt.savefig('gradient-descent-convergence.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved gradient-descent-convergence.png")

def generate_decision_boundaries():
    """Generate decision boundary plots for all three classifiers."""
    print("Generating decision boundary plots...")

    X, y = load_data()

    # Reduce to 2D with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    print(f"  Variance explained by 2 PCs: {pca.explained_variance_ratio_.sum():.1%}")

    # Initialize classifiers
    classifiers = [
        ('Logistic Regression', LogisticRegression(max_iter=1000)),
        ('KNN (k=5)', KNeighborsClassifier(n_neighbors=5)),
        ('Naive Bayes', GaussianNB())
    ]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (name, clf) in zip(axes, classifiers):
        # Train on 2D data
        clf.fit(X_pca, y)

        # Plot decision boundary
        DecisionBoundaryDisplay.from_estimator(
            clf, X_pca, response_method="predict",
            alpha=0.4, ax=ax, cmap=plt.cm.RdYlBu
        )

        # Plot data points
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y,
                            edgecolor="k", cmap=plt.cm.RdYlBu, s=30)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("PCA Component 1", fontsize=10)
        ax.set_ylabel("PCA Component 2", fontsize=10)

    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=plt.cm.RdYlBu(i/1),
                         markersize=8, markeredgecolor='k')
              for i in [0, 1]]
    labels = ['Class 0', 'Class 1']
    axes[-1].legend(handles, labels, title="Labels", loc="upper right")

    plt.tight_layout()
    plt.savefig('decision-boundaries.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved decision-boundaries.png")

def generate_pca_scatter():
    """Generate PCA scatter plot showing class separation."""
    print("Generating PCA scatter plot...")

    X, y = load_data()

    # Reduce to 2D with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))

    for label, color, marker in [(0, 'blue', 'o'), (1, 'red', 's')]:
        mask = y == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                  c=color, marker=marker, alpha=0.6,
                  label=f'Class {int(label)}', edgecolor='k', s=40)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('Marriage Dataset: PCA Projection', fontsize=14)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('pca-scatter.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved pca-scatter.png")

def generate_classifier_comparison():
    """Generate bar chart comparing classifier accuracies."""
    print("Generating classifier comparison plot...")

    X, y = load_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    # Train and evaluate
    classifiers = {
        'Logistic\nRegression': LogisticRegression(max_iter=1000),
        'KNN\n(k=5)': KNeighborsClassifier(n_neighbors=5),
        'Naive\nBayes': GaussianNB()
    }

    accuracies = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        accuracies[name] = acc
        print(f"  {name.replace(chr(10), ' ')}: {acc:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(accuracies.keys(), accuracies.values(),
                  color=['steelblue', 'darkorange', 'forestgreen'],
                  edgecolor='black')

    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Classifier Comparison on Marriage Dataset', fontsize=14)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f'{acc:.1%}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig('classifier-comparison.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved classifier-comparison.png")

def generate_naive_bayes_illustration():
    """Generate illustration of Naive Bayes decision process."""
    print("Generating Naive Bayes illustration...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Prior probabilities
    ax = axes[0]
    priors = {'Positive': 3/7, 'Negative': 4/7}
    colors = ['forestgreen', 'crimson']
    bars = ax.bar(priors.keys(), priors.values(), color=colors, edgecolor='black')
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Class Prior Probabilities', fontsize=12)
    ax.set_ylim(0, 0.7)
    for bar, val in zip(bars, priors.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.2f}', ha='center', va='bottom', fontsize=11)

    # Right: Word probabilities for a few words
    ax = axes[1]
    words = ['love', 'experience', 'hate', 'terrible']
    pos_probs = [1/13, 1/13, 0, 0]
    neg_probs = [0, 1/15, 1/15, 1/15]

    x = np.arange(len(words))
    width = 0.35

    bars1 = ax.bar(x - width/2, pos_probs, width, label='Positive',
                   color='forestgreen', edgecolor='black')
    bars2 = ax.bar(x + width/2, neg_probs, width, label='Negative',
                   color='crimson', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(words)
    ax.set_ylabel('P(word | class)', fontsize=12)
    ax.set_title('Word Probabilities by Class', fontsize=12)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('naive-bayes-illustration.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved naive-bayes-illustration.png")

def main():
    print("=" * 50)
    print("Generating images for Classification post")
    print("=" * 50)

    generate_sigmoid_plot()
    generate_gradient_descent_convergence()
    generate_pca_scatter()
    generate_classifier_comparison()
    generate_decision_boundaries()
    generate_naive_bayes_illustration()

    print("\n" + "=" * 50)
    print("All images generated successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()
