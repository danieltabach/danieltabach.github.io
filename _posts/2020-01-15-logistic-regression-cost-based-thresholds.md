---
layout: single
title: "Logistic Regression: When Accuracy Isn't Enough"
date: 2020-01-15
categories: [practical-ml]
tags: [r, classification, logistic-regression, roc-curve, threshold-optimization]
author_profile: true
toc: true
toc_label: "Contents"
toc_sticky: true
---

Most tutorials stop at accuracy. But what if misclassifying a customer costs you $50, while missing a fraudster costs $500?

This post goes beyond the basics. We'll build a logistic regression model, understand ROC curves, and then do what most tutorials skip: pick a threshold based on real business costs.

---

## The Dataset

We're using the [German Credit dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data), a classic for credit risk modeling. Each row represents a loan applicant, and the target variable tells us whether they defaulted (1) or not (0).

```r
# Load the data
germancredit <- read.table("germancredit.txt", header=FALSE, sep=" ")
data <- germancredit

# Rename columns for clarity
# Suffix convention: _C = categorical, _I = integer, _B = binary
new_names <- c("existing_checking_C", "Duration_I", "Credit_hist_C", "Purpose_C",
               "Cred_Amt_I", "Savings_C", "Employed_Dur_C", "Rate_I", "Status_Sex_C",
               "Debt_C", "Residence_I", "Property_C", "Age_I", "Installment_Plans_C",
               "Housing_C", "Open_Credit_I", "Job_C", "Dependents_I", "Telephone_B",
               "International_Employee_B", "Target")
colnames(data) <- new_names

# Clean up
data <- unique(data)  # Remove duplicates

# Quick null check - this dataset is clean, but always verify
null_count <- sapply(data, function(x) sum(is.na(x)))
# All zeros - we're good
```

---

## Quick EDA

The dataset has 20 features: a mix of categorical variables (credit history, purpose of loan, employment status) and numerical ones (age, loan duration, credit amount). The target is originally coded as 1/2, so we'll fix that.

```r
library(caret)

# Check for collinearity among numeric features
int_cols <- grep("_I$", colnames(data), value = TRUE)
correlation_matrix <- cor(data[, int_cols], use = "pairwise.complete.obs")
highly_correlated <- findCorrelation(correlation_matrix, cutoff = 0.75)
# No highly correlated features - good to go
```

---

## Preprocessing

Before modeling, we need to one-hot encode categorical variables. The key is to drop one dummy per category to avoid the multicollinearity trap.

```r
library(fastDummies)

cat_cols <- grep("_C$", colnames(data), value = TRUE)

# One-hot encode, dropping first dummy to prevent collinearity
data_encoded <- dummy_cols(data,
                           select_columns = cat_cols,
                           remove_first_dummy = TRUE,
                           remove_selected_columns = TRUE)

# Fix binary columns (they're stored as characters)
data_encoded$Telephone_B <- ifelse(data_encoded$Telephone_B == "A192", 1, 0)
data_encoded$International_Employee_B <- ifelse(data_encoded$International_Employee_B == "A201", 1, 0)

# Fix target: originally 1=good, 2=bad. Convert to 0/1
data_encoded$Target <- data_encoded$Target - 1
```

<details>
<summary><strong>Why drop one dummy variable?</strong></summary>

<p>If you have 3 categories (Red, Green, Blue), you only need 2 dummy columns. Why? Because if Red=0 and Green=0, you know it must be Blue.</p>

<p>Keeping all 3 creates perfect multicollinearity: the third column is always determined by the first two. This breaks the math behind regression.</p>

| Color | Red | Green | Blue (redundant) |
|-------|-----|-------|------------------|
| Red   | 1   | 0     | 0                |
| Green | 0   | 1     | 0                |
| Blue  | 0   | 0     | 1                |

<p>With just Red and Green columns, Blue is implied when both are 0.</p>

</details>

---

## Train-Test Split

```r
library(caTools)

set.seed(123)  # For reproducibility
splitIndex <- sample.split(data_encoded$Target, SplitRatio = 0.7)

train_data <- data_encoded[splitIndex, ]
validation_data <- data_encoded[!splitIndex, ]
```

---

## Building the Model

```r
# Fit logistic regression
# family = binomial tells R we're doing logistic (not linear) regression
logistic_model <- glm(Target ~ ., data = train_data, family = binomial)

# Get predicted probabilities on validation set
predicted_probs <- predict(logistic_model, newdata = validation_data, type = "response")
```

The model outputs probabilities between 0 and 1. To make actual predictions, we need a threshold: if probability > threshold, predict 1, otherwise predict 0.

The default threshold is 0.5. But is that optimal?

---

## The ROC Curve

The ROC curve shows every possible tradeoff between catching true positives and avoiding false positives. Each point on the curve represents a different threshold.

```r
library(pROC)

roc_obj <- roc(validation_data$Target, predicted_probs)
plot(roc_obj, main = "ROC Curve", col = "#1c61b6", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")  # Random classifier baseline
```

**How to read this:**
- The diagonal line represents random guessing (AUC = 0.5)
- A perfect classifier hugs the top-left corner (AUC = 1.0)
- Our curve falls somewhere in between

The **Area Under the Curve (AUC)** summarizes performance in one number. Higher is better.

<details>
<summary><strong>See it with a tiny example</strong></summary>

<p>Imagine 4 predictions with these probabilities and true labels:</p>

| Sample | Prob | True Label |
|--------|------|------------|
| A      | 0.9  | 1          |
| B      | 0.7  | 1          |
| C      | 0.4  | 0          |
| D      | 0.2  | 0          |

<p>At threshold = 0.5:</p>
<ul>
<li>Predict 1 for A, B (both correct = 2 True Positives)</li>
<li>Predict 0 for C, D (both correct = 2 True Negatives)</li>
</ul>
<p>Perfect! But what if threshold = 0.8?</p>
<ul>
<li>Only A gets predicted as 1</li>
<li>B becomes a False Negative (we missed it)</li>
</ul>

<p>The ROC curve plots all these tradeoffs as you sweep the threshold from 0 to 1.</p>

</details>

---

## Finding the Optimal Threshold: Youden's Index

If you want a balanced tradeoff between sensitivity and specificity, Youden's Index helps. It finds the threshold that maximizes: `sensitivity + specificity - 1`

```r
# Extract all points on the ROC curve
full_coords <- coords(roc_obj, "all")
specificities <- full_coords$specificity
sensitivities <- full_coords$sensitivity
thresholds <- full_coords$threshold

# Calculate Youden's Index for each threshold
youden_indices <- sensitivities + specificities - 1

# Find the threshold with the highest Youden's Index
optimal_index <- which.max(youden_indices)
optimal_threshold <- thresholds[optimal_index]

# Visualize
plot(1 - specificities, sensitivities, type = "l",
     xlab = "1 - Specificity (False Positive Rate)",
     ylab = "Sensitivity (True Positive Rate)",
     main = "ROC Curve with Optimal Threshold")
points(1 - specificities[optimal_index], sensitivities[optimal_index],
       col = "red", pch = 19, cex = 1.5)
```

The red dot marks the "optimal" threshold for balanced performance.

---

## The Confusion Matrix

Using our optimal threshold:

```r
predicted_classes <- ifelse(predicted_probs > optimal_threshold, 1, 0)
confusion_matrix <- table(Predicted = predicted_classes, Actual = validation_data$Target)
print(confusion_matrix)
```

<details>
<summary><strong>See it with a tiny example</strong></summary>

<p>A confusion matrix for 100 predictions might look like:</p>

|                | Actual 0 | Actual 1 |
|----------------|----------|----------|
| **Predicted 0**| 154 (TN) | 56 (FN)  |
| **Predicted 1**| 30 (FP)  | 60 (TP)  |

<p><strong>Reading it:</strong></p>
<ul>
<li><strong>True Negatives (154)</strong>: Correctly predicted no default</li>
<li><strong>False Positives (30)</strong>: Predicted default, but they paid (we denied a good customer)</li>
<li><strong>False Negatives (56)</strong>: Predicted no default, but they did (we approved a bad customer)</li>
<li><strong>True Positives (60)</strong>: Correctly predicted default</li>
</ul>

<p><strong>Metrics from this matrix:</strong></p>
<ul>
<li>Accuracy = (154 + 60) / 300 = 71.3%</li>
<li>Precision = 60 / (60 + 30) = 66.7%</li>
<li>Recall = 60 / (60 + 56) = 51.7%</li>
</ul>

</details>

---

## When Accuracy Isn't Enough: Cost-Based Thresholds

Here's where it gets real. Youden's Index assumes false positives and false negatives are equally bad. They rarely are.

**Healthcare example:**
- False Positive (healthy person diagnosed with cancer): Unnecessary stress, tests, maybe surgery
- False Negative (sick person cleared as healthy): Disease progresses, potentially fatal

In healthcare, you'd rather have more false positives than miss a single cancer case. You want high recall, even at the cost of precision.

**Fraud detection example:**
- False Positive (legitimate transaction blocked): Annoyed customer, maybe lost sale
- False Negative (fraud goes through): Direct financial loss, chargebacks, fees

The costs depend on your business. A $10 false positive and a $1000 false negative require very different thresholds.

---

## Building a Cost-Based Optimizer

Let's find the threshold that minimizes total cost, not just maximizes some abstract metric.

```r
# Define your costs
# In this example: FP costs $1.10, FN costs $1.20
FP_Cost <- 1.1
FN_Cost <- 1.2

# Test every threshold from 0 to 1
threshold_seq <- seq(0.01, 0.99, 0.01)

# Store results
costs_df <- data.frame(
  threshold = numeric(length(threshold_seq)),
  cost = numeric(length(threshold_seq)),
  TP = numeric(length(threshold_seq)),
  TN = numeric(length(threshold_seq)),
  FP = numeric(length(threshold_seq)),
  FN = numeric(length(threshold_seq))
)

# Loop through thresholds
for (i in seq_along(threshold_seq)) {
  threshold <- threshold_seq[i]

  # Classify at this threshold
  predicted_classes <- ifelse(predicted_probs > threshold, 1, 0)

  # Build confusion matrix
  confusion_matrix <- table(Predicted = predicted_classes,
                            Actual = validation_data$Target)

  # Handle edge cases where all predictions are 0 or all are 1
  if (nrow(confusion_matrix) != 2 || ncol(confusion_matrix) != 2) {
    next
  }

  # Extract values
  TN <- confusion_matrix[1, 1]
  FP <- confusion_matrix[1, 2]
  FN <- confusion_matrix[2, 1]
  TP <- confusion_matrix[2, 2]

  # Calculate cost
  current_cost <- FN * FN_Cost + FP * FP_Cost

  # Store
  costs_df[i, ] <- c(threshold, current_cost, TP, TN, FP, FN)
}

# Remove empty rows
costs_df <- costs_df[costs_df$threshold != 0, ]
```

**What happens if you change the costs?** If FN_Cost is much higher than FP_Cost (like in healthcare), the optimal threshold shifts lower. The model becomes more aggressive about predicting positives to avoid missing any. If FP_Cost is higher (like in spam filtering where false positives annoy users), the threshold shifts higher.

---

## Finding the Cost-Optimal Threshold

```r
# Find minimum cost
min_cost_row <- costs_df[which.min(costs_df$cost), ]
optimal_threshold_cost <- min_cost_row$threshold

# Visualize the cost curve
plot(costs_df$threshold, costs_df$cost,
     type = "l",
     xlab = "Threshold",
     ylab = "Total Cost",
     main = "Total Cost vs. Threshold")
abline(v = optimal_threshold_cost, col = "red", lty = 2)
text(optimal_threshold_cost + 0.05, max(costs_df$cost) * 0.9,
     paste("Optimal:", optimal_threshold_cost), col = "red")
```

With our example costs (FP = 1.1, FN = 1.2), the optimal threshold lands around 0.6. This is higher than the default 0.5 because false negatives are slightly more expensive, so we want to be more confident before predicting "no default."

---

## When Would You Use This?

| Scenario | FP Cost | FN Cost | Threshold Strategy |
|----------|---------|---------|-------------------|
| Cancer screening | Low | Very High | Low threshold (catch everyone) |
| Spam filter | High | Low | High threshold (don't block real email) |
| Fraud detection | Medium | High | Lower threshold (catch more fraud) |
| Loan approval | Medium | Medium | Use Youden's or 0.5 |

The key insight: **the "best" threshold depends entirely on your problem**. A model with 95% accuracy might be useless if it's missing the 5% that costs you the most.

---

## Summary

1. **Logistic regression outputs probabilities**, not classes. You need a threshold to convert them.

2. **ROC curves show all possible thresholds** and their tradeoffs between true positive rate and false positive rate.

3. **Youden's Index finds a balanced threshold**, but assumes equal costs for both types of errors.

4. **Cost-based optimization** finds the threshold that minimizes actual business cost. This is what you should use in production.

5. **Always define your costs first**. Talk to stakeholders. What does a false positive cost? A false negative? The math is easy once you know the numbers.

---

## Full Code

<details>
<summary><strong>Complete R Script</strong></summary>

```r
# Load packages
library(caTools)
library(fastDummies)
library(caret)
library(pROC)

# Load and prep data
germancredit <- read.table("germancredit.txt", header=FALSE, sep=" ")
data <- germancredit

new_names <- c("existing_checking_C", "Duration_I", "Credit_hist_C", "Purpose_C",
               "Cred_Amt_I", "Savings_C", "Employed_Dur_C", "Rate_I", "Status_Sex_C",
               "Debt_C", "Residence_I", "Property_C", "Age_I", "Installment_Plans_C",
               "Housing_C", "Open_Credit_I", "Job_C", "Dependents_I", "Telephone_B",
               "International_Employee_B", "Target")
colnames(data) <- new_names
data <- unique(data)

# Encode
cat_cols <- grep("_C$", colnames(data), value = TRUE)
data_encoded <- dummy_cols(data, select_columns = cat_cols,
                           remove_first_dummy = TRUE, remove_selected_columns = TRUE)
data_encoded$Telephone_B <- ifelse(data_encoded$Telephone_B == "A192", 1, 0)
data_encoded$International_Employee_B <- ifelse(data_encoded$International_Employee_B == "A201", 1, 0)
data_encoded$Target <- data_encoded$Target - 1

# Split
set.seed(123)
splitIndex <- sample.split(data_encoded$Target, SplitRatio = 0.7)
train_data <- data_encoded[splitIndex, ]
validation_data <- data_encoded[!splitIndex, ]

# Model
logistic_model <- glm(Target ~ ., data = train_data, family = binomial)
predicted_probs <- predict(logistic_model, newdata = validation_data, type = "response")

# ROC
roc_obj <- roc(validation_data$Target, predicted_probs)

# Cost optimization
FP_Cost <- 1.1
FN_Cost <- 1.2
threshold_seq <- seq(0.01, 0.99, 0.01)

costs_df <- data.frame(threshold = numeric(), cost = numeric())

for (threshold in threshold_seq) {
  predicted_classes <- ifelse(predicted_probs > threshold, 1, 0)
  cm <- table(Predicted = predicted_classes, Actual = validation_data$Target)

  if (nrow(cm) == 2 && ncol(cm) == 2) {
    FP <- cm[1, 2]
    FN <- cm[2, 1]
    cost <- FN * FN_Cost + FP * FP_Cost
    costs_df <- rbind(costs_df, data.frame(threshold = threshold, cost = cost))
  }
}

optimal_threshold <- costs_df$threshold[which.min(costs_df$cost)]
print(paste("Optimal threshold:", optimal_threshold))
```

</details>

---

*This tutorial is a remaster of work I did in early 2020, before I joined my first data science role. I've cleaned up the code and explanations, but kept the core analysis intact.*
