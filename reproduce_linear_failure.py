
import numpy as np
from sklearn.linear_model import LinearRegression

# Test case
import kalkulator_pkg.function_manager
from kalkulator_pkg.function_manager import find_function_from_data

print(f"Using function_manager from: {kalkulator_pkg.function_manager.__file__}")

data_points = [([1], 11), ([2], 12), ([3], 13)]
param_names = ['x']

print("Running find_function_from_data...")
result = find_function_from_data(data_points, param_names)
print("Result:", result)

# Manual check
X_vals = [1.0, 2.0, 3.0]
y_vals = [11.0, 12.0, 13.0]
X_arr = np.array(X_vals).reshape(-1, 1)
y_arr = np.array(y_vals)
lr = LinearRegression()
lr.fit(X_arr, y_arr)
y_pred = lr.predict(X_arr)
mse = np.mean((y_arr - y_pred) ** 2)
print("Manual Regression Check:")
print(f"Coef: {lr.coef_}, Intercept: {lr.intercept_}")
print(f"MSE: {mse}")
