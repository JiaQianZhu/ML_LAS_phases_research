"""
Author:JiaQian Zhu
Date:2023-12-6
Usage:This module provides a method for conducting experiments using the MLP Classifier with 5-fold cross-validation.
"""


import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, precision_score, recall_score

data = pd.read_csv('data/merged_data_normalized.csv', index_col=False)

X = data.iloc[:, 0:29]
y = data.iloc[:, 29:]

kf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_accuracies = []
all_hamming_losses = []
all_f1_scores = []
all_precisions = []
all_recalls = []

# Perform 5-fold cross-validation
for train_index, test_index in kf.split(X, y):
    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]

    # Create MLP model
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=10, activation='relu', random_state=42)

    # Fit the model
    mlp.fit(X_train, y_train)

    # Predict on the validation set
    y_pred_val = mlp.predict(X_val)

    # Calculate evaluation metrics for the fold
    accuracy = accuracy_score(y_val, y_pred_val)
    hamming = hamming_loss(y_val, y_pred_val)
    f1 = f1_score(y_val, y_pred_val, average='micro')
    precision = precision_score(y_val, y_pred_val, average='micro')
    recall = recall_score(y_val, y_pred_val, average='micro')

    # Store metrics for the fold
    all_accuracies.append(accuracy)
    all_hamming_losses.append(hamming)
    all_f1_scores.append(f1)
    all_precisions.append(precision)
    all_recalls.append(recall)

# Calculate standard deviations
std_accuracy = np.std(all_accuracies)
std_hamming = np.std(all_hamming_losses)
std_f1 = np.std(all_f1_scores)
std_precision = np.std(all_precisions)
std_recall = np.std(all_recalls)

# Print the average metrics across folds
print("Average Accuracy:", sum(all_accuracies) / len(all_accuracies))
print("Average Hamming Loss:", sum(all_hamming_losses) / len(all_hamming_losses))
print("Average F1 Score (micro):", sum(all_f1_scores) / len(all_f1_scores))
print("Average Precision (micro):", sum(all_precisions) / len(all_precisions))
print("Average Recall (micro):", sum(all_recalls) / len(all_recalls))

# Print standard deviations
print("Standard Deviation of Accuracy:", std_accuracy)
print("Standard Deviation of Hamming Loss:", std_hamming)
print("Standard Deviation of F1 Score (micro):", std_f1)
print("Standard Deviation of Precision (micro):", std_precision)
print("Standard Deviation of Recall (micro):", std_recall)
