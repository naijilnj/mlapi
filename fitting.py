#Train ↑ Test ↓  → Overfit
#Train ↓ Test ↓  → Underfit
#Train ↑ Test ↑  → Good model


import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Reproducibility
np.random.seed(42)

# Generate data
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.3, size=x.shape)

# Reshape for sklearn
x_reshaped = x.reshape(-1, 1)

# -------------------------------
# Simple Linear Regression
# -------------------------------
model = LinearRegression()
model.fit(x_reshaped, y)

y_pred = model.predict(x_reshaped)

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='red', label='data')
plt.plot(x, y_pred, color='blue', label='linear fit')

plt.title("Proper Fitting without polynomial")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()


# -------------------------------
# Polynomial Regression (Overfitting)
# -------------------------------
model_overfit = make_pipeline(
    PolynomialFeatures(degree=15),
    LinearRegression()
)

model_overfit.fit(x_reshaped, y)

y_pred_overfit = model_overfit.predict(x_reshaped)

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='red', label='data')
plt.plot(x, y_pred_overfit, color='blue', label='overfit')

plt.title("Overfitting")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()