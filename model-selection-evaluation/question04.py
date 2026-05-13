#Compare a model's performance at two different probability thresholds (0.3 and 0.7). 
# Print the confusion matrix and F1 score at each threshold. Explain the tradeoff.

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import numpy as np
# Load data
data = load_breast_cancer()
X = data.data
y = data.target
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
# Train model
pipeline.fit(X_train, y_train)
# Get predicted probabilities
y_probs = pipeline.predict_proba(X_test)[:, 1]
# Thresholds to evaluate
thresholds = [0.3, 0.7]
for thresh in thresholds:
    print(f"\nThreshold: {thresh}")
    y_pred = (y_probs >= thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    f1 = f1_score(y_test, y_pred)
    print(f"F1 Score: {f1:.2f}")
    
# Tradeoff explanation:
# A lower threshold (0.3) will classify more samples as positive, increasing recall but potentially lowering precision due to more false positives.
# A higher threshold (0.7) will classify fewer samples as positive, increasing precision but
# potentially lowering recall due to more false negatives. The F1 score helps balance this tradeoff by considering both precision and recall.
    
