import pandas as pd
import numpy as np

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("data.csv")

# -------------------------------
# Basic Info
# -------------------------------
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

# -------------------------------
# Missing Values
# -------------------------------
print(df.isnull().sum())

# Drop missing values
df_drop = df.dropna()

# Fill with constant
df_fill0 = df.fillna(0)

# Forward Fill
df_ffill = df.ffill()

# Backward Fill
df_bfill = df.bfill()

# Fill with mean / median / mode
df_mean = df.fillna(df.mean(numeric_only=True))
df_median = df.fillna(df.median(numeric_only=True))
df_mode = df.fillna(df.mode().iloc[0])

# -------------------------------
# Column-wise Fill
# -------------------------------
df['col_name'] = df['col_name'].fillna(df['col_name'].mean())

# -------------------------------
# Remove Duplicates
# -------------------------------
df = df.drop_duplicates()

# -------------------------------
# Encoding (Categorical → Numeric)
# -------------------------------
# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category'] = le.fit_transform(df['category'])

# One-Hot Encoding
df = pd.get_dummies(df, columns=['category'])

# -------------------------------
# Feature Scaling
# -------------------------------
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standard Scaling (mean=0, std=1)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.select_dtypes(include=np.number))

# Min-Max Scaling (0 to 1)
minmax = MinMaxScaler()
df_minmax = minmax.fit_transform(df.select_dtypes(include=np.number))

# -------------------------------
# Train-Test Split
# -------------------------------
from sklearn.model_selection import train_test_split

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Sklearn Pipeline (Imputation + Scaling)
# -------------------------------
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

num_cols = X.select_dtypes(include=['int64', 'float64']).columns

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols)
])

X_processed = preprocessor.fit_transform(X)