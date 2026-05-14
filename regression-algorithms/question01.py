#Train a Linear Regression model on the diabetes dataset. 
#Print the coefficients for each feature
#identify the top 3 most influential features, and compute RMSE.


from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error
import numpy as np
X, y = load_diabetes(return_X_y=True)
feature_names = load_diabetes().feature_names
print("Feature names:", feature_names)

model = LinearRegression()
model.fit(X, y)

print("Coefficients:", model.coef_)

# Identify the top 3 most influential features

feature_importance = np.abs(model.coef_)
top_3_indices = np.argsort(feature_importance)[-3:]
top_3_features = [feature_names[i] for i in top_3_indices]
print("Top 3 most influential features:", top_3_features)

# Compute RMSE
y_pred = model.predict(X)
rmse = root_mean_squared_error(y, y_pred)
print("RMSE:", rmse)
