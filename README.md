# Delhivery Graph-Based ETA Intelligence

End-to-end data science project for optimizing logistics ETA prediction using graph-based network intelligence.

## What this project includes

- Directed graph construction from trip segments
- Bottleneck hub and delay corridor audit
- Baseline ETA model vs graph-enhanced ETA model
- FTL vs Carting decision framework with time-cost tradeoff
- Strategy memo template for Network Operations leadership
- Optional Streamlit dashboard

## Project structure

- `data/raw/` input datasets (CSV/Parquet)
- `data/processed/` generated features and graph summaries
- `src/pipeline/` data preparation
- `src/graph/` graph construction and metrics
- `src/models/` baseline and graph-enhanced models
- `src/decision/` route-type recommendation framework
- `src/reporting/` memo and impact calculations
- `scripts/` executable pipeline runners
- `app/` optional Streamlit dashboard

## Expected input schema

Trip-level data file (example: `data/raw/trips.csv`) with at least:

- `shipment_id`
- `source_facility`
- `dest_facility`
- `departure_ts`
- `arrival_ts`
- `route_type` (FTL/Carting)
- `osrm_eta_minutes`
- `actual_transit_minutes` (or derivable from departure and arrival)
- `promised_eta_minutes` (optional but recommended)
- `distance_km` (optional but recommended)

## Quick start

1. Create environment and install dependencies:

   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`

2. Put your input file at `data/raw/trips.csv`

3. Run full pipeline:

   - `python scripts/run_full_pipeline.py`

4. View outputs in:

   - `outputs/metrics/`
   - `outputs/figures/`
   - `outputs/tables/`
   - `reports/network_operations_strategy_memo.md`

5. Optional dashboard:
   - `streamlit run app/streamlit_app.py`

## Business metrics tracked

- MAE on ETA prediction
- `% within 15%` ETA accuracy metric
- SLA breach contribution by hub and corridor
- Simulated late-delivery reduction for top-3 hub upgrades

