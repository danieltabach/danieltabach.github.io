"""
Generate images for the Logistic Regression Cost-Based Thresholds blog post.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

# Output directory
OUTPUT_DIR = "."

def load_german_credit():
    """Load German Credit dataset from UCI repository."""
    try:
        from ucimlrepo import fetch_ucirepo
        german_credit = fetch_ucirepo(id=144)
        X = german_credit.data.features
        y = german_credit.data.targets
        # Flatten target if needed
        if hasattr(y, 'values'):
            y = y.values.ravel()
        # Convert to binary (1 = good credit, 2 = bad credit) -> (0 = good, 1 = bad)
        y = (y == 2).astype(int)
        return X, y
    except ImportError:
        print("ucimlrepo not installed. Using synthetic data for demo.")
        # Create synthetic data similar to credit risk
        np.random.seed(42)
        n_samples = 1000
        X = pd.DataFrame({
            'duration': np.random.randint(6, 72, n_samples),
            'credit_amount': np.random.randint(250, 20000, n_samples),
            'age': np.random.randint(19, 75, n_samples),
            'num_credits': np.random.randint(1, 5, n_samples),
        })
        # Create target correlated with features
        prob = 1 / (1 + np.exp(-(0.01 * X['duration'] + 0.0001 * X['credit_amount'] - 0.02 * X['age'])))
        y = (np.random.random(n_samples) < prob).astype(int)
        return X, y


def preprocess_data(X, y):
    """Preprocess the data for modeling."""
    # Encode categorical variables
    X_processed = X.copy()
    for col in X_processed.columns:
        if X_processed[col].dtype == 'object':
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))

    # Handle missing values
    X_processed = X_processed.fillna(X_processed.median())

    return X_processed, y


def generate_roc_curve():
    """Generate ROC curve plot."""
    print("Generating ROC curve plot...")

    X, y = load_german_credit()
    X_processed, y = preprocess_data(X, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.3, random_state=42
    )

    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Get predictions
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color='#1c61b6', lw=2,
            label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',
            label='Random Classifier')

    ax.fill_between(fpr, tpr, alpha=0.2, color='#1c61b6')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title('ROC Curve: German Credit Risk Model', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/roc-curve.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved roc-curve.png")

    return X_train, X_test, y_train, y_test, model, y_proba, thresholds


def generate_youden_plot(X_test, y_test, y_proba):
    """Generate plot showing Youden's Index optimal threshold."""
    print("Generating Youden's Index plot...")

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    # Calculate Youden's Index
    youden = tpr - fpr
    optimal_idx = np.argmax(youden)
    optimal_threshold = thresholds[optimal_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color='#1c61b6', lw=2, label='ROC Curve')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

    # Mark optimal point
    ax.scatter([fpr[optimal_idx]], [tpr[optimal_idx]],
               color='red', s=150, zorder=5,
               label=f'Optimal (Youden\'s J)\nThreshold = {optimal_threshold:.3f}')

    # Draw lines to point
    ax.plot([fpr[optimal_idx], fpr[optimal_idx]], [0, tpr[optimal_idx]],
            'r--', alpha=0.5)
    ax.plot([0, fpr[optimal_idx]], [tpr[optimal_idx], tpr[optimal_idx]],
            'r--', alpha=0.5)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Finding the Optimal Threshold with Youden\'s Index', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/youden-optimal.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved youden-optimal.png")

    return optimal_threshold


def generate_confusion_matrix_plot(y_test, y_proba, threshold):
    """Generate confusion matrix visualization."""
    print("Generating confusion matrix plot...")

    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom", fontsize=11)

    # Labels
    classes = ['Good Credit (0)', 'Bad Credit (1)']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted Label',
           ylabel='True Label',
           title=f'Confusion Matrix (Threshold = {threshold:.3f})')

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            label = f'{cm[i, j]}'
            if i == 0 and j == 0:
                label += '\n(TN)'
            elif i == 0 and j == 1:
                label += '\n(FP)'
            elif i == 1 and j == 0:
                label += '\n(FN)'
            else:
                label += '\n(TP)'
            ax.text(j, i, label, ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/confusion-matrix.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved confusion-matrix.png")


def generate_cost_curve(X_test, y_test, y_proba):
    """Generate cost curve for different thresholds."""
    print("Generating cost curve plot...")

    # Define costs
    FP_cost = 1.0  # Cost of false positive (deny good customer)
    FN_cost = 5.0  # Cost of false negative (approve bad customer who defaults)

    thresholds = np.arange(0.1, 0.9, 0.01)
    costs = []
    fps = []
    fns = []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        cm = confusion_matrix(y_test, y_pred)

        # Handle different confusion matrix shapes
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            continue

        total_cost = fp * FP_cost + fn * FN_cost
        costs.append(total_cost)
        fps.append(fp)
        fns.append(fn)

    thresholds = thresholds[:len(costs)]
    costs = np.array(costs)

    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(thresholds, costs, 'b-', lw=2, label='Total Cost')
    ax.axvline(x=optimal_threshold, color='red', linestyle='--', lw=2,
               label=f'Optimal Threshold = {optimal_threshold:.2f}')
    ax.scatter([optimal_threshold], [costs[optimal_idx]],
               color='red', s=100, zorder=5)

    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Total Cost', fontsize=12)
    ax.set_title(f'Cost Curve (FP Cost = ${FP_cost:.0f}, FN Cost = ${FN_cost:.0f})', fontsize=14)
    ax.legend(loc='upper right', fontsize=11)

    # Add annotation
    ax.annotate(f'Minimum Cost\nat threshold {optimal_threshold:.2f}',
                xy=(optimal_threshold, costs[optimal_idx]),
                xytext=(optimal_threshold + 0.1, costs[optimal_idx] + 20),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='black'))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/cost-curve.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved cost-curve.png")

    return optimal_threshold


def generate_threshold_comparison():
    """Generate comparison of different threshold strategies."""
    print("Generating threshold comparison plot...")

    X, y = load_german_credit()
    X_processed, y = preprocess_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics at different thresholds
    thresholds = [0.3, 0.5, 0.7]
    metrics = []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            # Cost (FP=$1, FN=$5)
            cost = fp * 1 + fn * 5

            metrics.append({
                'threshold': thresh,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cost': cost,
                'fp': fp,
                'fn': fn
            })

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Metrics comparison
    ax = axes[0]
    x = np.arange(len(thresholds))
    width = 0.2

    ax.bar(x - 1.5*width, [m['accuracy'] for m in metrics], width, label='Accuracy', color='#1f77b4')
    ax.bar(x - 0.5*width, [m['precision'] for m in metrics], width, label='Precision', color='#ff7f0e')
    ax.bar(x + 0.5*width, [m['recall'] for m in metrics], width, label='Recall', color='#2ca02c')
    ax.bar(x + 1.5*width, [m['f1'] for m in metrics], width, label='F1 Score', color='#d62728')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_title('Classification Metrics by Threshold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in thresholds])
    ax.legend()
    ax.set_ylim(0, 1)

    # Right: Cost comparison
    ax = axes[1]
    colors = ['#4CAF50', '#FFC107', '#F44336']
    bars = ax.bar(x, [m['cost'] for m in metrics], color=colors, edgecolor='black')

    ax.set_ylabel('Total Cost ($)', fontsize=12)
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_title('Business Cost by Threshold (FP=$1, FN=$5)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in thresholds])

    # Add value labels on bars
    for bar, m in zip(bars, metrics):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'${int(height)}\n(FP={m["fp"]}, FN={m["fn"]})',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/threshold-comparison.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved threshold-comparison.png")


def generate_cost_scenarios():
    """Generate visualization of different cost scenarios."""
    print("Generating cost scenarios plot...")

    X, y = load_german_credit()
    X_processed, y = preprocess_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Different cost scenarios
    scenarios = [
        {'name': 'Equal Costs\n(FP=$1, FN=$1)', 'fp_cost': 1, 'fn_cost': 1, 'color': '#1f77b4'},
        {'name': 'Fraud Detection\n(FP=$1, FN=$10)', 'fp_cost': 1, 'fn_cost': 10, 'color': '#ff7f0e'},
        {'name': 'Spam Filter\n(FP=$10, FN=$1)', 'fp_cost': 10, 'fn_cost': 1, 'color': '#2ca02c'},
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    thresholds = np.arange(0.1, 0.9, 0.01)

    for scenario in scenarios:
        costs = []
        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                cost = fp * scenario['fp_cost'] + fn * scenario['fn_cost']
                costs.append(cost)
            else:
                costs.append(np.nan)

        costs = np.array(costs)
        valid_mask = ~np.isnan(costs)

        # Normalize for comparison
        costs_norm = (costs - np.nanmin(costs)) / (np.nanmax(costs) - np.nanmin(costs) + 1e-10)

        optimal_idx = np.nanargmin(costs)

        ax.plot(thresholds[valid_mask], costs_norm[valid_mask],
                lw=2, label=scenario['name'], color=scenario['color'])
        ax.scatter([thresholds[optimal_idx]], [costs_norm[optimal_idx]],
                   s=100, color=scenario['color'], zorder=5, edgecolor='black')

    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Normalized Cost', fontsize=12)
    ax.set_title('Optimal Threshold Shifts Based on Cost Structure', fontsize=14)
    ax.legend(loc='upper center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/cost-scenarios.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved cost-scenarios.png")


def main():
    print("=" * 50)
    print("Generating images for Logistic Regression post")
    print("=" * 50)

    # Generate ROC curve and get data
    X_train, X_test, y_train, y_test, model, y_proba, thresholds = generate_roc_curve()

    # Generate Youden's Index plot
    optimal_threshold = generate_youden_plot(X_test, y_test, y_proba)

    # Generate confusion matrix
    generate_confusion_matrix_plot(y_test, y_proba, optimal_threshold)

    # Generate cost curve
    generate_cost_curve(X_test, y_test, y_proba)

    # Generate threshold comparison
    generate_threshold_comparison()

    # Generate cost scenarios
    generate_cost_scenarios()

    print("\n" + "=" * 50)
    print("All images generated successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
