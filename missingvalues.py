import pandas as pd
import numpy as np

# -------------------------------
# Create Dataset with Missing Values
# -------------------------------

data = {
    'First Score': [100, 90, np.nan, 95, 75, 87],
    'Second Score': [30, 45, 56, np.nan, 60, 70],
    'Third Score': [np.nan, 40, 80, 98, 55, np.nan]
}

df = pd.DataFrame(data)

print(df)


# -------------------------------
# Drop Missing Values
# -------------------------------
print("\nAfter dropping missing values:")
print(df.dropna())


# -------------------------------
# Fill Missing Values with 0
# -------------------------------
df2 = df.fillna(0)
print("\nFill NA with 0:")
print(df2)


# -------------------------------
# Mean Imputation
# -------------------------------
print("\nColumn Means:")
print(df.mean())

df3 = df.fillna(df.mean())

print("\nImputation using mean:")
print(df3)


# -------------------------------
# Median Imputation
# -------------------------------
df4 = df.fillna(df.median())

print("\nImputation using median:")
print(df4)


# -------------------------------
# Forward Fill & Backward Fill
# -------------------------------
df5 = df.ffill()
print("\nForward Fill:")
print(df5)

df6 = df.bfill()
print("\nBackward Fill:")
print(df6)


# -------------------------------
# Sklearn Imputation + Scaling Pipeline
# -------------------------------
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Select numerical columns
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns

# Pipeline for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler())
])

# Column Transformer
preprocessor_pipeline = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features)
    ]
)

# Apply transformation
df_data = preprocessor_pipeline.fit_transform(df)

print("\nProcessed Data:")
print(df_data)