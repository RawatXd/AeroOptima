import pandas as pd
import joblib
import pulp


def score_flights(origin, date, dep_hour_min, dep_hour_max,
                   model_path='models/xgboost_delay_model.pkl',
                   encoders_path='models/label_encoders.pkl',
                   data_path='DataSet/flight_data_features.csv'):
    """Load trained XGBoost model and score flights for a given airport/date/window."""
    xgb_model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)

    df = pd.read_csv(data_path)
    subset = df[df['ORIGIN'] == origin].copy()
    subset = subset[(subset['FL_DATE'] == date) &
                     (subset['DEP_HOUR'] >= dep_hour_min) &
                     (subset['DEP_HOUR'] < dep_hour_max)].copy()

    features = ['MONTH', 'DAY_OF_WEEK', 'DEP_HOUR', 'IS_PEAK_HOUR', 'IS_WEEKEND',
                'ORIGIN_DAILY_FLIGHTS', 'ROUTE_AVG_DELAY', 'CARRIER_AVG_DELAY',
                'OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST']
    X = subset[features].copy()

    categorical_cols = ['OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST']
    for col in categorical_cols:
        le = encoders[col]
        class_to_idx = {cls: idx for idx, cls in enumerate(le.classes_)}
        unknown_idx = len(le.classes_)
        X[col] = X[col].map(lambda x: class_to_idx.get(x, unknown_idx))

    subset['DELAY_PROBABILITY'] = xgb_model.predict_proba(X)[:, 1]
    return subset.reset_index(drop=True)


def _prepare_gate_windows(flights_df, boarding_time, turnaround_buffer, slot_size):
    """Shared helper: compute gate-occupancy windows and time slots for a flight set."""

    def hhmm_to_minutes(t):
        return (t // 100) * 60 + (t % 100)

    flights_df = flights_df.copy()
    flights_df['DEP_MIN'] = flights_df['CRS_DEP_TIME'].apply(hhmm_to_minutes)
    flights_df['GATE_START'] = flights_df['DEP_MIN'] - boarding_time
    flights_df['GATE_END'] = flights_df['DEP_MIN'] + turnaround_buffer

    window_start = int(flights_df['GATE_START'].min())
    window_end = int(flights_df['GATE_END'].max())
    time_slots = list(range(window_start, window_end, slot_size))

    flight_ids = flights_df.index.tolist()
    flight_slots = {
        i: [t for t in time_slots
            if flights_df.loc[i, 'GATE_START'] <= t < flights_df.loc[i, 'GATE_END']]
        for i in flight_ids
    }

    slot_occupancy = {t: 0 for t in time_slots}
    for i in flight_ids:
        for t in flight_slots[i]:
            slot_occupancy[t] += 1
    max_concurrent = max(slot_occupancy.values())

    return flights_df, flight_ids, time_slots, flight_slots, max_concurrent


def optimize_gates(flights_df, boarding_time=40, turnaround_buffer=30,
                    slot_size=5, gate_buffer=5):
    """Run the time-indexed, risk-aware gate assignment optimization."""

    flights_df, flight_ids, time_slots, flight_slots, max_concurrent = _prepare_gate_windows(
        flights_df, boarding_time, turnaround_buffer, slot_size
    )

    num_gates = max_concurrent + gate_buffer
    gate_ids = list(range(num_gates))

    model = pulp.LpProblem("Gate_Assignment_Risk_Aware", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("assign", (flight_ids, gate_ids), cat='Binary')

    for i in flight_ids:
        model += pulp.lpSum(x[i][j] for j in gate_ids) == 1

    for j in gate_ids:
        for t in time_slots:
            flights_in_slot = [i for i in flight_ids if t in flight_slots[i]]
            if len(flights_in_slot) > 1:
                model += pulp.lpSum(x[i][j] for i in flights_in_slot) <= 1

    model += pulp.lpSum(
        flights_df.loc[i, 'DELAY_PROBABILITY'] * j * x[i][j]
        for i in flight_ids for j in gate_ids
    )

    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)
    status = pulp.LpStatus[model.status]

    assignment = []
    if status == 'Optimal':
        for i in flight_ids:
            for j in gate_ids:
                if x[i][j].value() == 1:
                    assignment.append({
                        'flight_idx': i,
                        'gate': j,
                        'delay_probability': flights_df.loc[i, 'DELAY_PROBABILITY']
                    })

    assignment_df = pd.DataFrame(assignment)
    correlation = (assignment_df['delay_probability'].corr(assignment_df['gate'])
                   if not assignment_df.empty else None)

    return {
        'status': status,
        'num_gates': num_gates,
        'max_concurrent': max_concurrent,
        'assignment': assignment_df,
        'correlation': correlation,
        'flights_df': flights_df
    }


def naive_gate_assignment(flights_df, boarding_time=40, turnaround_buffer=30,
                           slot_size=5, gate_buffer=5):
    """Baseline: assign gates first-come-first-served by departure time, no risk awareness."""

    flights_df, flight_ids, time_slots, flight_slots, max_concurrent = _prepare_gate_windows(
        flights_df, boarding_time, turnaround_buffer, slot_size
    )

    num_gates = max_concurrent + gate_buffer
    flight_ids_sorted = flights_df.sort_values('DEP_MIN').index.tolist()

    gate_occupied_slots = {j: set() for j in range(num_gates)}
    assignment = []

    for i in flight_ids_sorted:
        this_flight_slots = set(flight_slots[i])
        assigned_gate = None
        for j in range(num_gates):
            if not (this_flight_slots & gate_occupied_slots[j]):
                assigned_gate = j
                gate_occupied_slots[j] |= this_flight_slots
                break
        assignment.append({
            'flight_idx': i,
            'gate': assigned_gate,
            'delay_probability': flights_df.loc[i, 'DELAY_PROBABILITY']
        })

    assignment_df = pd.DataFrame(assignment)
    correlation = (assignment_df['delay_probability'].corr(assignment_df['gate'])
                   if not assignment_df.empty else None)

    return {
        'num_gates': num_gates,
        'max_concurrent': max_concurrent,
        'assignment': assignment_df,
        'correlation': correlation
    }


if __name__ == '__main__':
    flights = score_flights(origin='ATL', date='2023-01-09',
                             dep_hour_min=6, dep_hour_max=12)
    print(f"Flights scored: {len(flights)}")

    result = optimize_gates(flights)
    naive = naive_gate_assignment(flights)

    print(f"\n--- Optimized (risk-aware) ---")
    print(f"Solver status: {result['status']}")
    print(f"Gates used: {result['num_gates']}")
    print(f"Correlation (delay prob vs gate index): {result['correlation']:.4f}")

    print(f"\n--- Naive (first-come-first-served) ---")
    print(f"Gates used: {naive['num_gates']}")
    print(f"Correlation (delay prob vs gate index): {naive['correlation']:.4f}")

    output = flights.merge(
        result['assignment'], left_index=True, right_on='flight_idx'
    )
    output.to_csv('DataSet/final_gate_assignment.csv', index=False)

    naive_output = flights.merge(
        naive['assignment'], left_index=True, right_on='flight_idx'
    )
    naive_output.to_csv('DataSet/naive_gate_assignment.csv', index=False)

    print("\nBoth results saved.")