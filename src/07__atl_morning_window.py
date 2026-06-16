import pandas as pd

df = pd.read_csv('DataSet/flight_data_features.csv')

atl = df[df['ORIGIN'] == 'ATL'].copy()
atl_morning = atl[(atl['FL_DATE'] == '2023-01-09') & (atl['DEP_HOUR'] >= 9) & (atl['DEP_HOUR'] < 10)]

print("Flights in this window:", atl_morning.shape[0])
print(atl_morning[['FL_DATE', 'OP_UNIQUE_CARRIER', 'CRS_DEP_TIME', 'CRS_ARR_TIME', 'DEP_HOUR']].head(10))
print("\nCarrier distribution:\n", atl_morning['OP_UNIQUE_CARRIER'].value_counts())