#predict the outcome of a binary classification model using the AUC-ROC metric.
from sklearn.metrics import roc_auc_score
import numpy as np
y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
y_pred_good = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.85, 0.3, 0.75])
y_pred_random = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
y_pred_bad = np.array([0.9, 0.8, 0.2, 0.1, 0.85, 0.15, 0.7, 0.25])
for name, preds in [("Good", y_pred_good),("Random", y_pred_random),("Bad (inverted)", y_pred_bad)]:
    auc = roc_auc_score(y_true, preds)
print(f"{name:<20}: AUC = {auc:.3f}")