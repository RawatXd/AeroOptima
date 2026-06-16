# ✈️ AeroOptima — Delay-Aware Gate Optimization

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://aerooptima-zxitmxh8t9adba2ya9t6eb.streamlit.app/)

![Demo](demo.gif)

AeroOptima predicts flight delay risk using Machine Learning, then uses Integer Programming to assign gates in a way that prioritizes high-risk flights — instead of the naive first-come-first-served approach airports typically default to.

## Why AeroOptima?

Most delay-prediction projects stop at a model and a metric. AeroOptima goes one step further: it turns a delay probability into an actual operational decision.

| | Naive Scheduling | AeroOptima |
|---|---|---|
| Delay awareness | None | XGBoost-predicted probability per flight |
| Gate assignment | First-come-first-served | Risk-aware Integer Program |
| Conflict handling | Manual / none | Turnaround-time constraints, solved optimally |
| Risk-to-gate correlation | ~0.02 (no relationship) | ~-0.65 (high-risk flights prioritized) |

## Architecture

```
Flight Schedule (BTS Data)
        │
        ▼
  XGBoost Model ──► Delay Probability per Flight
        │
        ▼
  Integer Program (PuLP)
   - One gate per flight
   - No turnaround conflicts
   - High-risk flights steered to priority gates
        │
        ▼
  Streamlit Dashboard
   - Optimized vs Naive comparison
   - Flight-level detail
```

The ML model never makes operational decisions — it only estimates risk. The optimizer never predicts anything — it only allocates gates under real constraints. Each module does one job well, and they connect through a single score: predicted delay probability.

## Results

- **Delay Prediction:** XGBoost achieves ROC-AUC of 0.71 using schedule-based features (departure hour, day of week, route history, carrier history). No live weather data is used, which sets a natural ceiling on this number.
- **Gate Optimization:** Solved to provable optimality for 319 flights at Atlanta (ATL), across a measured minimum of 94 gates — gate count is derived from actual peak concurrent demand in the data, not an assumed value.
- **Risk-Awareness Comparison:** The optimized assignment shows a -0.65 correlation between delay probability and gate index (high-risk flights get priority gates), versus 0.02 for naive first-come-first-served scheduling.

## A note on what "gate optimization" means here

Real airport gate data (physical layout, distances, capacities) isn't publicly available, so this project uses a simulated airport: a measured number of identical gates, sized to the real peak demand found in the data. The optimizer doesn't know which physical gate is "better" — it designates a subset of gate indices as priority slots and proves it can reliably steer high-risk flights into them while keeping every assignment turnaround-conflict-free. That's a real, solvable scheduling problem, presented honestly rather than dressed up as more than it is.

## Tech Stack

| Component | Technology |
|---|---|
| Delay Prediction | XGBoost, Scikit-Learn |
| Optimization | PuLP (CBC solver) |
| Data Processing | Pandas |
| Dashboard | Streamlit |
| Data Source | BTS On-Time Performance (US DOT) |

## Project Structure

```
AeroOptima/
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── src/
│   ├── 01_data_inspection.py
│   ├── 02_data_cleaning.py
│   ├── 03_feature_engineering.py
│   ├── 04_train_test_split.py
│   ├── 05_baseline_models.py
│   ├── 06_xg_boost.py
│   ├── 07_find_busiest_airport.py
│   ├── 07__atl_morning_window.py
│   ├── 08_gate_optimization.py
│   ├── 09_slack_analysis.py
│   └── run_pipeline.py         # Consolidated, callable pipeline
├── models/
│   ├── xgboost_delay_model.pkl
│   └── label_encoders.pkl
├── DataSet/
│   └── flight_data_features.csv
├── requirements.txt
└── README.md
```

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/AeroOptima.git
cd AeroOptima
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the dashboard**
```bash
streamlit run dashboard/app.py
```

## Data Source

[BTS On-Time Performance Data](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr) — U.S. Bureau of Transportation Statistics.

---

Built by [Your Name](https://github.com/<your-username>)
