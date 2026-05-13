# Demonstrate overfitting vs underfitting using a Decision Tree. 
# Train trees with depths 1, 3, 10, and None. Print train/test accuracy 
# for each and label each as underfitting, good fit, or overfitting.

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier
depths = [1, 3, 10, None]
for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    if depth == 1:
        fit_label = "Underfitting"
    elif depth == 3:
        fit_label = "Good Fit"
    else:
        fit_label = "Overfitting"
    print(f"Depth: {depth}, Train Acc: {train_acc:.2f}, Test Acc: {test_acc:.2f} - {fit_label}")    
    