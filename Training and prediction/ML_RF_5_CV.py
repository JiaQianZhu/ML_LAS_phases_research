"""
Author:JiaQian Zhu
Date:2023-12-6
Usage:This module provides a method for conducting experiments using the  Multi-label RF with 5-fold cross-validation.
"""
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, precision_score, recall_score, \
    label_ranking_loss, coverage_error, zero_one_loss, make_scorer
import pandas as pd


data = pd.read_csv('data/merged_data_normalized.csv', index_col=False)


X = data.iloc[:, 0:29]
y = data.iloc[:, 29:]


metrics = {
    'accuracy': make_scorer(accuracy_score),
    'hamming_loss': make_scorer(hamming_loss),
    'f1_micro': make_scorer(f1_score, average='micro'),
    'precision_micro': make_scorer(precision_score, average='micro'),
    'recall_micro': make_scorer(recall_score, average='micro'),
}

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

multi_output_rf = MultiOutputClassifier(rf_classifier, n_jobs=-1)

cv_predictions = cross_val_predict(multi_output_rf, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42))


accuracy = accuracy_score(y, cv_predictions)
hamming_loss_value = hamming_loss(y, cv_predictions)
f1_score_value = f1_score(y, cv_predictions, average='micro')
precision_score_value = precision_score(y, cv_predictions, average='micro')
recall_score_value = recall_score(y, cv_predictions, average='micro')
label_ranking_loss_value = label_ranking_loss(y, cv_predictions)
coverage_error_value = coverage_error(y, cv_predictions)
zero_one_loss_value = zero_one_loss(y, cv_predictions)


print("5-CV results:")
print(f"Accuracy: {accuracy}")
print(f"Hamming Loss: {hamming_loss_value}")
print(f"F1 Score (micro): {f1_score_value}")
print(f"Precision (micro): {precision_score_value}")
print(f"Recall (micro): {recall_score_value}")
print(f"Label Ranking Loss: {label_ranking_loss_value}")
print(f"Coverage Error: {coverage_error_value}")
print(f"One Error: {zero_one_loss_value}")