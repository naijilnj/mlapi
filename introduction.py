# Experiment 1 - Introduction

import numpy as np

# -------------------------------
# Basic Array
# -------------------------------
array = [20, 25, 90, 95, 100, 105, 110, 115, 120, 125]

# -------------------------------
# NumPy Operations
# -------------------------------
A = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
]

print("A =", A)

# First row
print(A[0])

# First row using NumPy
print(np.array(A)[0, :])

# Column extraction
k = 2
res = np.array(A)[:, k]
print(list(res))

# Row extraction
res2 = np.array(A)[k, :]
print(res2)


# -------------------------------
# Pandas Operations
# -------------------------------
import pandas as pd

d = {
    'col1': [1, 2, 3, 4, 5, 7, 8],
    'col2': [4, 5, 6, 9, 5, 7, 8],
    'col3': [7, 8, 12, 1, 11, 4, 8]
}

df = pd.DataFrame(data=d)

print(df)

# Shape & size
print(df.shape)
num_of_rows = len(df)
print(num_of_rows)
print(df.shape[0])
print(df.index.size)

print(len(df.columns))
print(df.shape[1])

# Head & Tail
print(df.head())
print(df.tail())
print(df.head(3))

# Columns
print(list(df.columns))

# Info
df.info()


# -------------------------------
# Statistical Functions
# -------------------------------
print(df.mean())
print(df.median())
print(df.std())
print(df.var())
print(df.min())
print(df.max())

# Column specific
print(df['col2'].mean())
print(df['col1'].median())


# -------------------------------
# Describe & Null Check
# -------------------------------
print(df.describe())

print(df.isnull())
print(df.notnull())


# -------------------------------
# Data with Missing Values
# -------------------------------
d2 = {
    'col1': [1, 2, 3, np.nan, 4],
    'col2': [5, np.nan, 6, 7, 8],
    'col3': [9, 10, 11, 12, np.nan]
}

df2 = pd.DataFrame(d2)

print(df2)

# Null checks
print(df2.isnull())
print(df2.notnull())

# Info
df2.info()

# Count null values
print(df2.isnull().sum())
print(df2.isnull().sum().sum())