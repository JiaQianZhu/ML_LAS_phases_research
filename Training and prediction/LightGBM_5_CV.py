"""
Author:JiaQian Zhu
Date:2023-12-6
Usage:This module provides a method for conducting experiments using the LightGBM with 5-fold cross-validation.
"""

import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.multioutput import MultiOutputClassifier
import lightgbm as lgb

from sklearn.metrics import hamming_loss
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

data = pd.read_csv('../data/merged_data_normalized.csv', index_col=False)

X = data.iloc[:, 0:29]
y = data.iloc[:, 29:]

mlb = MultiLabelBinarizer()
y_binary = mlb.fit_transform(y.values)

lgb_classifier = lgb.LGBMClassifier(objective='binary', random_state=42)

multioutput_xgb_classifier = MultiOutputClassifier(lgb_classifier)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

hamming_loss_values = []

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    multioutput_xgb_classifier.fit(X_train, y_train)
    y_pred_fold = multioutput_xgb_classifier.predict(X_test)

    hamming_loss_fold = hamming_loss(y_test, y_pred_fold)
    hamming_loss_values.append(hamming_loss_fold)


accuracy_values = cross_val_score(multioutput_xgb_classifier, X, y, cv=kf, scoring='accuracy')
f1_score_values = cross_val_score(multioutput_xgb_classifier, X, y, cv=kf, scoring='f1_micro')
precision_score_values = cross_val_score(multioutput_xgb_classifier, X, y, cv=kf, scoring='precision_micro')
recall_score_values = cross_val_score(multioutput_xgb_classifier, X, y, cv=kf, scoring='recall_micro')


print("LightGBM_5_CV Results:")
print("\nMean and Standard Deviation for Each Metric:")
print(f"Hamming Loss: Mean = {np.mean(hamming_loss_values)}, Std = {np.std(hamming_loss_values)}")
print(f"Accuracy: Mean = {np.mean(accuracy_values)}, Std = {np.std(accuracy_values)}")
print(f"F1 Score (micro): Mean = {np.mean(f1_score_values)}, Std = {np.std(f1_score_values)}")
print(f"Precision (micro): Mean = {np.mean(precision_score_values)}, Std = {np.std(precision_score_values)}")
print(f"Recall (micro): Mean = {np.mean(recall_score_values)}, Std = {np.std(recall_score_values)}")





