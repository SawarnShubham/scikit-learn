# What do the confusion matrix numbers mean? 
# Compute precision, recall, and F1 manually from the matrix.

from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score
import numpy as np
y_true = np.array([1,0,1,1,0,1,0,0,1,1])
y_pred = np.array([1,0,0,1,0,1,1,0,1,0])
cm = confusion_matrix(y_true, y_pred)
print(cm)
tn, fp, fn, tp = cm.ravel()
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)
print("Precision:", precision)
print("Recall:", recall)    
print("F1 Score:", f1)
