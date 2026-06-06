import numpy as np
import time
from kalkulator_pkg.symbolic_regression.genetic_engine import GeneticSymbolicRegressor
from kalkulator_pkg.symbolic_regression.genetic_config import GeneticConfig

config = GeneticConfig(
    population_size=100, 
    generations=20, 
    timeout=30,
    operators=["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "pow", "sqrt", "lambertw", "abs", "neg", "tanh", "max", "min", "square", "cube"]
)

x = np.linspace(-5, 5, 100)
X = x.reshape(-1, 1)

print("="*50)
print("Test 2: Nested Composition (f(x) = ln(abs(x) + sin(x^2)))")
y2 = np.log(np.abs(x) + np.sin(x**2))
reg2 = GeneticSymbolicRegressor(config)
reg2.fit(X, y2, variable_names=["x"])
pred2 = reg2.predict(X)
mse2 = np.mean(np.abs(y2 - pred2)**2)
print("Result:", reg2.get_expression())
print(f"MSE: {mse2:.6g}")

print("\n" + "="*50)
print("Test 3: Broken Heuristic (f(x) = max(0, x) * cos(x))")
y3 = np.maximum(0, x) * np.cos(x)
reg3 = GeneticSymbolicRegressor(config)
reg3.fit(X, y3, variable_names=["x"])
pred3 = reg3.predict(X)
mse3 = np.mean(np.abs(y3 - pred3)**2)
print("Result:", reg3.get_expression())
print(f"MSE: {mse3:.6g}")

print("\n" + "="*50)
print("Test 4: Kill Shot (f(x) = sin(x) * max(0, cos(x)))")
y4 = np.sin(x) * np.maximum(0, np.cos(x))
reg4 = GeneticSymbolicRegressor(config)
reg4.fit(X, y4, variable_names=["x"])
pred4 = reg4.predict(X)
mse4 = np.mean(np.abs(y4 - pred4)**2)
print("Result:", reg4.get_expression())
print(f"MSE: {mse4:.6g}")
print("="*50)
