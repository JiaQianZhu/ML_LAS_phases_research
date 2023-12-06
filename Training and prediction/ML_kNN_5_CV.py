"""
Author:JiaQian Zhu
Date:2023-12-6
Usage:This module provides a method for conducting experiments using the  Multi-label kNN with 5-fold cross-validation.
"""
import pandas as pd

data = pd.read_csv('data/merged_data_normalized.csv', index_col=False)

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, precision_score, recall_score

from skmultilearn.adapt import MLkNN
import warnings
from scipy.sparse import csr_matrix

warnings.filterwarnings('ignore')

X = data.iloc[:, 0:29]
y = data.iloc[:, 29:]

classifier = MLkNN()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("kNN_5_CV Results:")

hamming_loss_values = []
accuracy_values = []
f1_score_values = []
precision_score_values = []
recall_score_values = []

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    y_train_sparse = csr_matrix(y_train.values)

    classifier.fit(np.array(X_train), y_train_sparse)
    y_pred_fold = classifier.predict(X_test)

    hamming_loss_fold = hamming_loss(y_test, y_pred_fold)
    hamming_loss_values.append(hamming_loss_fold)

    accuracy_fold = accuracy_score(y_test, y_pred_fold)
    accuracy_values.append(accuracy_fold)

    f1_score_fold = f1_score(y_test, y_pred_fold, average='micro')
    f1_score_values.append(f1_score_fold)

    precision_score_fold = precision_score(y_test, y_pred_fold, average='micro')
    precision_score_values.append(precision_score_fold)

    recall_score_fold = recall_score(y_test, y_pred_fold, average='micro')
    recall_score_values.append(recall_score_fold)

print(f"Hamming Loss: Mean = {np.mean(hamming_loss_values)}, Std = {np.std(hamming_loss_values)}")
print(f"Accuracy: Mean = {np.mean(accuracy_values)}, Std = {np.std(accuracy_values)}")
print(f"f1_scores(micro): Mean = {np.mean(f1_score_values)}, Std = {np.std(f1_score_values)}")
print(f"precision(micro): Mean = {np.mean(precision_score_values)}, Std = {np.std(precision_score_values)}")
print(f"recall(micro): Mean = {np.mean(recall_score_values)}, Std = {np.std(recall_score_values)}")
