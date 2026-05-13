#You train a model and notice train accuracy=0.99 and test accuracy=0.72. 
# Explain the problem and demonstrate two fixes that reduce the gap.

from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
X, y = make_classification(n_samples=300, n_features=20, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a deep tree (overfitting)
model = DecisionTreeClassifier(max_depth=None, random_state=42)
model.fit(X_train, y_train) 
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Overfitting Tree - Train Acc: {train_acc:.2f}, Test Acc: {test_acc:.2f}")
# Fix 1: Limit tree depth   
model_limited = DecisionTreeClassifier(max_depth=3, random_state=42)
model_limited.fit(X_train, y_train)
train_acc_limited = model_limited.score(X_train, y_train)
test_acc_limited = model_limited.score(X_test, y_test)
print(f"Limited Depth Tree - Train Acc: {train_acc_limited:.2f}, Test Acc: {test_acc_limited:.2f}")
# Fix 2: Use Random Forest (ensemble method)
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
train_acc_rf = model_rf.score(X_train, y_train)
test_acc_rf = model_rf.score(X_test, y_test)
print(f"Random Forest - Train Acc: {train_acc_rf:.2f}, Test Acc: {test_acc_rf:.2f}")
