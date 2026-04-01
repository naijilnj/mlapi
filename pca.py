import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Generate Random Dataset
# -------------------------------
np.random.seed(42)

X = np.random.randint(100, size=(100, 3))
X_dataset = pd.DataFrame(X)

print(X_dataset.head())


# -------------------------------
# Standardization
# -------------------------------
scaler = StandardScaler()

X_std = scaler.fit_transform(X_dataset)

print("Mean:", np.mean(X_std))
print("Std Dev:", np.std(X_std), "\n")

X_std_dataset = pd.DataFrame(X_std)
print(X_std_dataset.head())


# -------------------------------
# PCA Transformation
# -------------------------------
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_std_dataset)

print("Shape after PCA:", X_pca.shape, "\n")
print("First 5 rows of PCA result:\n", X_pca[:5])


# -------------------------------
# Explained Variance
# -------------------------------
explained_var_ratio = pca.explained_variance_ratio_

print("Explained Variance Ratio:", explained_var_ratio)

cumulative_var_ratio = np.cumsum(explained_var_ratio)

print("Cumulative Variance:", cumulative_var_ratio)


# -------------------------------
# Plot Cumulative Variance
# -------------------------------
plt.plot(
    range(1, len(cumulative_var_ratio) + 1),
    cumulative_var_ratio,
    marker='o'
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance Ratio")
plt.title("Cumulative Variance Ratio vs Number of Principal Components")

plt.show()