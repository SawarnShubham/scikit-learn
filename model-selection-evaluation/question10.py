#Use cross_val_score to compare 5 different models on the same dataset using the same CV folds.
# Display results sorted by mean score and identify the best model.

from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load data
X, y = load_wine(return_X_y=True)

# SAME folds for every model
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf"))
    ])
}

results = {}

for name, model in models.items():
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )
    results[name] = scores

sorted_results = sorted(
    results.items(),
    key=lambda x: x[1].mean(),
    reverse=True
)

print("Model Performance:\n")

for name, scores in sorted_results:
    print(
        f"{name}: "
        f"Mean={scores.mean():.4f}, "
        f"Std={scores.std():.4f}"
    )

best_model = sorted_results[0][0]

print(f"\nBest Model: {best_model}")