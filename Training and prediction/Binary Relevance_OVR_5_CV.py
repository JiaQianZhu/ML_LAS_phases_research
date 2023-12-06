"""
Author:JiaQian Zhu
Date:2023-12-6
Usage:This module provides a method for conducting experiments using the Binary Relevance with 5-fold cross-validation.
"""
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss
import pandas as pd


data = pd.read_csv('data/merged_data_normalized.csv', index_col=False)

X = data.iloc[:, 0:29]
y = data.iloc[:, 29:]


classifier = RandomForestClassifier()

multi_label_classifier = OneVsRestClassifier(classifier)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("Binary Relevance_5_CV Results:")

hamming_loss_values = []

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    multi_label_classifier.fit(X_train, y_train)
    y_pred_fold = multi_label_classifier.predict(X_test)

    hamming_loss_fold = hamming_loss(y_test, y_pred_fold)
    hamming_loss_values.append(hamming_loss_fold)

accuracy_values = cross_val_score(multi_label_classifier, X, y, cv=kf, scoring='accuracy')
f1_score_values = cross_val_score(multi_label_classifier, X, y, cv=kf, scoring='f1_micro')
precision_score_values = cross_val_score(multi_label_classifier, X, y, cv=kf, scoring='precision_micro')
recall_score_values = cross_val_score(multi_label_classifier, X, y, cv=kf, scoring='recall_micro')

# Print the mean and standard deviation for each metric
print("\nMean and Standard Deviation for Each Metric:")
print(f"Hamming Loss: Mean = {np.mean(hamming_loss_values)}, Std = {np.std(hamming_loss_values)}")
print(f"Accuracy: Mean = {np.mean(accuracy_values)}, Std = {np.std(accuracy_values)}")
print(f"F1 Score (micro): Mean = {np.mean(f1_score_values)}, Std = {np.std(f1_score_values)}")
print(f"Precision (micro): Mean = {np.mean(precision_score_values)}, Std = {np.std(precision_score_values)}")
print(f"Recall (micro): Mean = {np.mean(recall_score_values)}, Std = {np.std(recall_score_values)}")
