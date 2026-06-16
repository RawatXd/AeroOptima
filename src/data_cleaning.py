import pandas as pd

# Load
df = pd.read_csv('DataSet/raw/flight-data.csv')

# Step 1: Drop cancelled and diverted flights
df = df[df['CANCELLED'] == 0]
df = df[df['DIVERTED'] == 0]

# Step 2: Drop rows where DEP_DELAY or ARR_DELAY is still missing
df = df.dropna(subset=['DEP_DELAY', 'ARR_DELAY'])

# Step 3: Fill delay cause columns with 0 (no delay = no cause)
delay_cause_cols = ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 
                    'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']
df[delay_cause_cols] = df[delay_cause_cols].fillna(0)

# Step 4: Remove extreme outliers in DEP_DELAY (beyond 6 hours = 360 mins)
# We keep them for now but flag it
print("Flights with DEP_DELAY > 360 mins:", (df['DEP_DELAY'] > 360).sum())

# Step 5: Create our target variable
# 1 = delayed by more than 15 minutes, 0 = not delayed
df['IS_DELAYED'] = (df['DEP_DELAY'] > 15).astype(int)

# Step 6: Check results
print("Shape after cleaning:", df.shape)
print("\nMissing values after cleaning:\n", df.isnull().sum())
print("\nDelay distribution:\n", df['IS_DELAYED'].value_counts())
print("\nDelay rate: {:.2%}".format(df['IS_DELAYED'].mean()))

# Step 7: Save cleaned data
df.to_csv('flight_data_cleaned.csv', index=False)
print("\nCleaned data saved.")
