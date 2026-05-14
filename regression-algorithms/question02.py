from sklearn.linear_model import Ridge
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Without scaling
ridge_raw = Ridge(alpha=1.0)
ridge_raw.fit(X_train, y_train)

print("WITHOUT scaling")
print("Coefficients:", ridge_raw.coef_.round(2))
print("R2:", ridge_raw.score(X_test, y_test))

# With scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge_scaled = Ridge(alpha=1.0)
ridge_scaled.fit(X_train_scaled, y_train)

print("\nWITH scaling")
print("Coefficients:", ridge_scaled.coef_.round(2))
print("R2:", ridge_scaled.score(X_test_scaled, y_test))
