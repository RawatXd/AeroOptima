import pandas as pd

df = pd.read_csv('DataSet/flight_data_features.csv')
print(df['ORIGIN'].value_counts().head(10))