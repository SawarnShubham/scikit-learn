from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

pipe.fit(X_train, y_train)

train_accuracy = pipe.score(X_train, y_train)
test_accuracy = pipe.score(X_test, y_test)

print("Train Accuracy:", round(train_accuracy, 3))
print("Test Accuracy:", round(test_accuracy, 3))

predictions = pipe.predict(X_test[:3])
probabilities = pipe.predict_proba(X_test[:3])

print("Predictions:", predictions)
print("Probabilities:\n", probabilities.round(3))