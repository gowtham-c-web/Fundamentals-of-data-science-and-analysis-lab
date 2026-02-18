import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
Y = np.array([2, 4, 5, 4, 5, 7, 8, 9, 9, 11])
X = X.reshape(-1, 1)
model = LinearRegression()
model.fit(X, Y)
slope = model.coef_[0]
intercept = model.intercept_
print(f"Slope (beta_1): {slope:.4f}")
print(f"Intercept (beta_0): {intercept:.4f}")
Y_pred = model.predict(X)
plt.figure(figsize=(8, 5))
plt.scatter(X, Y, color='blue', alpha=0.7, label='Actual Data')
plt.plot(X, Y_pred, color='red', linewidth=2, label='Fitted Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simple Linear Regression: Fitted Line')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
r_squared = model.score(X, Y)
print(f"R-squared: {r_squared:.4f}")
X_new = np.array([[15]])
Y_new = model.predict(X_new)
print(f"Predicted Y for X = 15: {Y_new[0]:.4f}")
