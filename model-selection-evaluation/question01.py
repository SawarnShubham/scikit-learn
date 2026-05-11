# Run 5-fold cross-validation on a RandomForestClassifier 
# using the breast cancer dataset. 
# Print the score for each fold, the mean, and the standard deviation. 
# Use stratified folds.


from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

rf = RandomForestClassifier(random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(rf, X, y, cv=skf, scoring='accuracy')

print("Cross-validation scores:", scores)
print("Mean:", scores.mean())
print("Standard deviation:", scores.std())  