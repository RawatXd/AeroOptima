import pandas as pd
import joblib
import pulp

# ============================================================
# PART 2 — Load model, score the ATL 9-10 AM flights
# ============================================================

xgb_model = joblib.load('models/xgboost_delay_model.pkl')
encoders = joblib.load('models/label_encoders.pkl')

df = pd.read_csv('DataSet/flight_data_features.csv')
atl = df[df['ORIGIN'] == 'ATL'].copy()
flights = atl[(atl['FL_DATE'] == '2023-01-09') & (atl['DEP_HOUR'] >= 6) & (atl['DEP_HOUR'] < 12)].copy()

features = ['MONTH', 'DAY_OF_WEEK', 'DEP_HOUR', 'IS_PEAK_HOUR', 'IS_WEEKEND',
            'ORIGIN_DAILY_FLIGHTS', 'ROUTE_AVG_DELAY', 'CARRIER_AVG_DELAY',
            'OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST']

X = flights[features].copy()

categorical_cols = ['OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST']
for col in categorical_cols:
    le = encoders[col]
    class_to_idx = {cls: idx for idx, cls in enumerate(le.classes_)}
    unknown_idx = len(le.classes_)
    X[col] = X[col].map(lambda x: class_to_idx.get(x, unknown_idx))

flights['DELAY_PROBABILITY'] = xgb_model.predict_proba(X)[:, 1]

print("Flights scored:", flights.shape[0])
print(flights[['FL_DATE', 'OP_UNIQUE_CARRIER', 'CRS_DEP_TIME', 'CRS_ARR_TIME', 'DELAY_PROBABILITY']].head(10))
print("\nDelay probability stats:\n", flights['DELAY_PROBABILITY'].describe())

flights.to_csv('DataSet/atl_morning_scored.csv', index=False)
print("\nScored flights saved.")

# ============================================================
# PART 3 — Gate assignment optimization (time-indexed formulation)
# ============================================================

flights_df = pd.read_csv('DataSet/atl_morning_scored.csv')

def hhmm_to_minutes(t):
    hours = t // 100
    minutes = t % 100
    return hours * 60 + minutes

flights_df['DEP_MIN'] = flights_df['CRS_DEP_TIME'].apply(hhmm_to_minutes)

BOARDING_TIME = 40
TURNAROUND_BUFFER = 30
flights_df['GATE_START'] = flights_df['DEP_MIN'] - BOARDING_TIME
flights_df['GATE_END'] = flights_df['DEP_MIN'] + TURNAROUND_BUFFER

flight_ids = flights_df.index.tolist()

# Discretize time into 5-minute slots covering the full range of gate occupancy
SLOT_SIZE = 5
window_start = int(flights_df['GATE_START'].min())
window_end = int(flights_df['GATE_END'].max())
time_slots = list(range(window_start, window_end, SLOT_SIZE))

# For each flight, determine which time slots it occupies
flight_slots = {}
for i in flight_ids:
    start = flights_df.loc[i, 'GATE_START']
    end = flights_df.loc[i, 'GATE_END']
    flight_slots[i] = [t for t in time_slots if start <= t < end]

# ---- DIAGNOSTIC: measure true peak concurrent gate demand ----
slot_occupancy = {t: 0 for t in time_slots}
for i in flight_ids:
    for t in flight_slots[i]:
        slot_occupancy[t] += 1

max_concurrent = max(slot_occupancy.values())
peak_slot = max(slot_occupancy, key=slot_occupancy.get)
print(f"\nMax concurrent flights needing a gate at once: {max_concurrent}")
print(f"This occurs at time slot: {peak_slot} minutes from midnight")
# ----------------------------------------------------------------

NUM_GATES = max_concurrent + 5  # measured minimum, plus small buffer
gate_ids = list(range(NUM_GATES))
print(f"Using NUM_GATES = {NUM_GATES}")

# Build the model
model = pulp.LpProblem("Gate_Assignment", pulp.LpMinimize)

x = pulp.LpVariable.dicts("assign", (flight_ids, gate_ids), cat='Binary')

# Constraint 1: every flight assigned to exactly one gate
for i in flight_ids:
    model += pulp.lpSum(x[i][j] for j in gate_ids) == 1

# Constraint 2: for every gate and every time slot, at most one flight occupies it
for j in gate_ids:
    for t in time_slots:
        flights_in_slot = [i for i in flight_ids if t in flight_slots[i]]
        if len(flights_in_slot) > 1:
            model += pulp.lpSum(x[i][j] for i in flights_in_slot) <= 1

# Objective: minimize total delay-risk exposure (placeholder, refine next)
model += pulp.lpSum(flights_df.loc[i, 'DELAY_PROBABILITY'] *
                     pulp.lpSum(x[i][j] for j in gate_ids)
                     for i in flight_ids)

solver = pulp.PULP_CBC_CMD(msg=True)
model.solve(solver)

print("\nSolver status:", pulp.LpStatus[model.status])
print("Number of time slots:", len(time_slots))
print("Number of binary variables:", len(flight_ids) * len(gate_ids))   