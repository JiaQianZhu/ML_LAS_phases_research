"""
Author:JiaQian Zhu
Date:2023-12-6
Usage:This module provides a method for conducting experiments using the DecisionTreeClassifier with 5-fold cross-validation.
"""

from sklearn.model_selection import cross_val_predict, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, precision_score, recall_score,make_scorer
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


tree_classifier = DecisionTreeClassifier(random_state=42)

multi_output_tree = MultiOutputClassifier(tree_classifier, n_jobs=-1)

cv_predictions = cross_val_predict(multi_output_tree, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42))


accuracy = accuracy_score(y, cv_predictions)
hamming_loss_value = hamming_loss(y, cv_predictions)
f1_score_value = f1_score(y, cv_predictions, average='micro')
precision_score_value = precision_score(y, cv_predictions, average='micro')
recall_score_value = recall_score(y, cv_predictions, average='micro')


print("5-CV results:")
print(f"Accuracy: {accuracy}")
print(f"Hamming Loss: {hamming_loss_value}")
print(f"F1 Score (micro): {f1_score_value}")
print(f"Precision (micro): {precision_score_value}")
print(f"Recall (micro): {recall_score_value}")

