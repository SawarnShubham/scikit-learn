# A model shows 99% accuracy on a dataset where 99% of samples are class 0.
# Show why accuracy is misleading here by printing the confusion matrix 
# and per-class F1 scores.

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
y_true = np.array([0]*990 + [1]*10) # 99% class 0
y_pred = np.array([0]*1000) # model always predicts 0
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1']))

