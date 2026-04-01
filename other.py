# =========================
# IMPORTS
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data.csv")

print(df.head())
print(df.info())
print(df.describe())

# =========================
# HANDLE MISSING VALUES
# =========================
df.fillna(df.mean(), inplace=True)
# df.dropna(inplace=True)  # alternative

# =========================
# SPLIT FEATURES & TARGET
# =========================
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# =========================
# FEATURE SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# LINEAR REGRESSION
# =========================
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)

print("\nLinear Regression Predictions:")
print(y_pred_lin)

# =========================
# LOGISTIC REGRESSION
# =========================
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("\nLogistic Regression Accuracy:")
print(accuracy_score(y_test, y_pred_log))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))

# =========================
# PCA
# =========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# =========================
# PCA PLOT
# =========================
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title("PCA Plot")
plt.show()

# =========================
# K-MEANS CLUSTERING
# =========================
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.labels_

print("\nK-Means Labels:")
print(labels)

# =========================
# SEARCH / FILTER
# =========================
# Example: change column_name accordingly
# result = df[df["column_name"] > 50]
# print(result)

# =========================
# SAVE OUTPUT
# =========================
df.to_csv("output.csv", index=False)

print("\nProcess Completed Successfully!")