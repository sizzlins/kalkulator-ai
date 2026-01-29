
import numpy as np

# User data
points = [
    (4.5, 0.618369803069737), (4.4, 0.655077897178518), (4.3, 0.695582793748033), 
    (4.2, 0.740176853196037), (4.1, 0.788945462844258), (4.0, 0.841470984807897), 
    (3.9, 0.896192201029956), (3.8, 0.948984619355586), (3.7, 0.989903076372124), 
    (3.6, 0.995407957751765), (3.5, 0.909297426825682), (3.4, 0.598472144103956), 
    (3.3, -0.190567962875487), (3.2, -0.95892427466314), (3.1, -0.544021110889362), 
    (2.9, 0.544021110889362), (2.8, 0.95892427466314), (2.7, 0.190567962875487), 
    (2.6, -0.598472144103956), (2.5, -0.909297426825682), (2.4, -0.995407957751765), 
    (2.3, -0.989903076372124), (2.2, -0.948984619355586), (2.1, -0.896192201029956), 
    (2.0, -0.841470984807897), (1.9, -0.788945462844257), (1.8, -0.740176853196037), 
    (1.7, -0.695582793748033), (1.6, -0.655077897178519), (1.5, -0.618369803069737), 
    (1.4, -0.585097272940462), (1.3, -0.554893791463713), (1.2, -0.527415385771866), 
    (1.1, -0.502351154603513), (1.0, -0.479425538604203)
]

X = np.array([p[0] for p in points])
y_true = np.array([p[1] for p in points])

# Target function: sin(1/(x-3))
y_target = np.sin(1 / (X - 3.0))

# Found function: sin(1/(x-3.02057))
y_found = np.sin(1 / (X - 3.02057))

mse_target = np.mean((y_true - y_target)**2)
mse_found = np.mean((y_true - y_found)**2)

print(f"MSE of Exact Target (3.0): {mse_target:.10f}")
print(f"MSE of Found Function (3.02): {mse_found:.10f}")
print(f"Diff: {mse_found - mse_target}")

# Detect why optimization failed
# Check gradient at 3.02057?
# Let's brute force scan around 3.0
shifts = np.linspace(2.9, 3.1, 1000)
mses = []
for s in shifts:
    y_s = np.sin(1 / (X - s))
    mses.append(np.mean((y_true - y_s)**2))

min_mse_scan = min(mses)
best_shift = shifts[np.argmin(mses)]
print(f"Best Scan MSE: {min_mse_scan:.10f} at shift={best_shift}")
