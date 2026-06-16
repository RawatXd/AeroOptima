import pandas as pd

# Load the data
df = pd.read_csv('DataSet/raw/flight-data.csv')  

# Basic inspection
print("Shape:", df.shape)
print("\nColumn Names:\n", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())
print("\nMissing Values:\n", df.isnull().sum())
print("\nBasic Stats:\n", df.describe())
