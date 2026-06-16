import pandas as pd

# Load cleaned data
df = pd.read_csv('DataSet/flight_data_cleaned.csv')

# 1. Convert FL_DATE to datetime
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])

# 2. Day of week (0=Monday, 6=Sunday)
df['DAY_OF_WEEK'] = df['FL_DATE'].dt.dayofweek

# 3. Extract hour from CRS_DEP_TIME
# CRS_DEP_TIME is in HHMM format (e.g. 1327 = 13:27), we just need the hour
df['DEP_HOUR'] = (df['CRS_DEP_TIME'] // 100).astype(int)

# Fix edge case: 2400 should become 0
df['DEP_HOUR'] = df['DEP_HOUR'].replace(24, 0)

# 4. Peak hour indicator (morning 7-9am, evening 5-8pm are typically congested)
df['IS_PEAK_HOUR'] = df['DEP_HOUR'].apply(
    lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 20) else 0
)

# 5. Weekend indicator
df['IS_WEEKEND'] = df['DAY_OF_WEEK'].apply(lambda x: 1 if x >= 5 else 0)

# 6. Airport congestion score — how many flights depart from this airport that day
airport_daily_counts = df.groupby(['ORIGIN', 'FL_DATE']).size().reset_index(name='ORIGIN_DAILY_FLIGHTS')
df = df.merge(airport_daily_counts, on=['ORIGIN', 'FL_DATE'], how='left')

# 7. Previous delay history — average delay for this route in this dataset
route_avg_delay = df.groupby(['ORIGIN', 'DEST'])['DEP_DELAY'].mean().reset_index(name='ROUTE_AVG_DELAY')
df = df.merge(route_avg_delay, on=['ORIGIN', 'DEST'], how='left')

# 8. Carrier average delay — some airlines are systematically worse
carrier_avg_delay = df.groupby('OP_UNIQUE_CARRIER')['DEP_DELAY'].mean().reset_index(name='CARRIER_AVG_DELAY')
df = df.merge(carrier_avg_delay, on='OP_UNIQUE_CARRIER', how='left')

# Check results
print("Shape after feature engineering:", df.shape)
print("\nNew columns:\n", df[['DAY_OF_WEEK', 'DEP_HOUR', 'IS_PEAK_HOUR', 
                                'IS_WEEKEND', 'ORIGIN_DAILY_FLIGHTS', 
                                'ROUTE_AVG_DELAY', 'CARRIER_AVG_DELAY']].head(10))
print("\nMissing values check:\n", df[['ORIGIN_DAILY_FLIGHTS', 'ROUTE_AVG_DELAY', 'CARRIER_AVG_DELAY']].isnull().sum())

# Save
df.to_csv('DataSet/flight_data_features.csv', index=False)
print("\nFeature-engineered data saved.")