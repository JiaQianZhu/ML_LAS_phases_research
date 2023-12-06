"""
Author:JiaQian Zhu
Date:2023-12-6
Usage:This module provides a method for conducting experiments using the  Classifier Chains with 5-fold cross-validation.
"""

import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import hamming_loss
import pandas as pd


data = pd.read_csv('data/merged_data_normalized.csv', index_col=False)

X = data.iloc[:, 0:29]
y = data.iloc[:, 29:]

base_classifier = RandomForestClassifier()


classifier_chain = ClassifierChain(base_classifier, order='random', random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

hamming_loss_values = []

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    classifier_chain.fit(X_train, y_train)
    y_pred_fold = classifier_chain.predict(X_test)

    hamming_loss_fold = hamming_loss(y_test, y_pred_fold)
    hamming_loss_values.append(hamming_loss_fold)


accuracy_values = cross_val_score(classifier_chain, X, y, cv=kf, scoring='accuracy')
f1_score_values = cross_val_score(classifier_chain, X, y, cv=kf, scoring='f1_micro')
precision_score_values = cross_val_score(classifier_chain, X, y, cv=kf, scoring='precision_micro')
recall_score_values = cross_val_score(classifier_chain, X, y, cv=kf, scoring='recall_micro')


print("Classifier chains_5_CV Results:")
print("\nMean and Standard Deviation for Each Metric:")
print(f"Hamming Loss: Mean = {np.mean(hamming_loss_values)}, Std = {np.std(hamming_loss_values)}")
print(f"Accuracy: Mean = {np.mean(accuracy_values)}, Std = {np.std(accuracy_values)}")
print(f"F1 Score (micro): Mean = {np.mean(f1_score_values)}, Std = {np.std(f1_score_values)}")
print(f"Precision (micro): Mean = {np.mean(precision_score_values)}, Std = {np.std(precision_score_values)}")
print(f"Recall (micro): Mean = {np.mean(recall_score_values)}, Std = {np.std(recall_score_values)}")
