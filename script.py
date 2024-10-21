# script.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate a sample dataset using sklearn's make_regression
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# 2. Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize the models
lr_model = LinearRegression()         # Linear Regression model
rf_model = RandomForestRegressor()    # Random Forest model

# 4. Train the models
lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# 5. Predict on the test data using both models
y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# 6. Calculate MSE and R-squared for both models

# Linear Regression Metrics
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Random Forest Metrics
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# 7. Print the results
print("Linear Regression:")
print(f"Mean Squared Error (MSE): {mse_lr}")
print(f"R-squared (R²): {r2_lr}\n")

print("Random Forest:")
print(f"Mean Squared Error (MSE): {mse_rf}")
print(f"R-squared (R²): {r2_rf}")


