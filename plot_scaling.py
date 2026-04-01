# Experiment 2

import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("diabetes.csv")

print(df.head())

# Basic info
print(df.shape)
df.info()

# Missing values
print(df.isnull().sum())
print(df['Pregnancies'].isnull().sum())

# Statistics
print(df.describe())
print(df.describe().T)


# -------------------------------
# Histogram
# -------------------------------
df.hist(bins=20, figsize=(15, 10), color='teal', edgecolor='black')

plt.suptitle("Histogram for each attribute")
plt.show()


# -------------------------------
# Boxplots (Subplots)
# -------------------------------
fig, axs = plt.subplots(len(df.columns), 1, dpi=95, figsize=(7, 17))

i = 0
for col in df.columns:
    axs[i].boxplot(df[col], vert=False)
    axs[i].set_ylabel(col)
    i += 1

plt.show()


# -------------------------------
# Single Boxplot
# -------------------------------
df.boxplot(vert=False)
plt.suptitle("Boxplot for each attribute")
plt.show()


# -------------------------------
# Correlation Heatmap
# -------------------------------
corr = df.corr()

sns.heatmap(corr, annot=True, fmt=".2f", cmap="crest")

plt.show()


# -------------------------------
# Pie Chart (Outcome)
# -------------------------------
plt.pie(
    df['Outcome'].value_counts(),
    labels=["Not Diabetes", "Diabetes"],
    autopct="%.2f"
)

plt.title("Outcome Proportionality")
plt.show()


# -------------------------------
# Feature & Target Split
# -------------------------------
X = df.drop(columns=['Outcome'])
y = df['Outcome']

print(X.head())


# -------------------------------
# MinMax Scaling
# -------------------------------
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)

print(rescaledX[:3])


# -------------------------------
# Standard Scaling
# -------------------------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)

rescaledX = scaler.transform(X)

print(rescaledX[:3])