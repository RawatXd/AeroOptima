import pandas as pd
from sklearn.model_selection import train_test_split

# Load feature-engineered data
df = pd.read_csv('DataSet/flight_data_features.csv')

# Drop the leaked features — we'll recalculate them properly below
df = df.drop(columns=['ROUTE_AVG_DELAY', 'CARRIER_AVG_DELAY'])

# Define target and features to use
target = 'IS_DELAYED'

# Split FIRST, before recalculating route/carrier averages
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df[target]
)

# Recalculate ROUTE_AVG_DELAY using ONLY training data
route_avg_delay = train_df.groupby(['ORIGIN', 'DEST'])['DEP_DELAY'].mean().reset_index(name='ROUTE_AVG_DELAY')
train_df = train_df.merge(route_avg_delay, on=['ORIGIN', 'DEST'], how='left')
test_df = test_df.merge(route_avg_delay, on=['ORIGIN', 'DEST'], how='left')

# Recalculate CARRIER_AVG_DELAY using ONLY training data
carrier_avg_delay = train_df.groupby('OP_UNIQUE_CARRIER')['DEP_DELAY'].mean().reset_index(name='CARRIER_AVG_DELAY')
train_df = train_df.merge(carrier_avg_delay, on='OP_UNIQUE_CARRIER', how='left')
test_df = test_df.merge(carrier_avg_delay, on='OP_UNIQUE_CARRIER', how='left')

# Check for NaNs in test set — happens if a route/carrier in test never appeared in train
print("NaNs in test ROUTE_AVG_DELAY:", test_df['ROUTE_AVG_DELAY'].isnull().sum())
print("NaNs in test CARRIER_AVG_DELAY:", test_df['CARRIER_AVG_DELAY'].isnull().sum())

# Fill any such NaNs with the global training mean (fallback for unseen routes)
global_mean_delay = train_df['DEP_DELAY'].mean()
train_df['ROUTE_AVG_DELAY'] = train_df['ROUTE_AVG_DELAY'].fillna(global_mean_delay)
test_df['ROUTE_AVG_DELAY'] = test_df['ROUTE_AVG_DELAY'].fillna(global_mean_delay)
test_df['CARRIER_AVG_DELAY'] = test_df['CARRIER_AVG_DELAY'].fillna(global_mean_delay)

print("\nTrain shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("\nTrain delay rate:", train_df[target].mean())
print("Test delay rate:", test_df[target].mean())

# Save
train_df.to_csv('DataSet/train.csv', index=False)
test_df.to_csv('DataSet/test.csv', index=False)
print("\nTrain/test sets saved.")