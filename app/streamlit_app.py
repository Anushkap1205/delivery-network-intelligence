from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
METRICS = ROOT / "outputs" / "metrics"
REPORT = ROOT / "reports" / "network_operations_strategy_memo.md"

st.set_page_config(page_title="Delhivery Network Intelligence", layout="wide")
st.title("Delhivery Graph-Based ETA Intelligence")

metrics_file = METRICS / "model_comparison.json"
if metrics_file.exists():
    with open(metrics_file, "r", encoding="utf-8") as f:
        m = json.load(f)
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline MAE", f'{m["baseline_mae"]:.2f}')
    c2.metric("Graph MAE", f'{m["graph_mae"]:.2f}', delta=f'{m["graph_advantage_mae_reduction"]:.2f}')
    c3.metric("Graph Acc@15%", f'{m["graph_acc15"]:.2%}', delta=f'{m["graph_advantage_acc15_delta"]:.2%}')
else:
    st.info("Run `python scripts/run_full_pipeline.py` to generate outputs.")

st.subheader("Top Bottleneck Hubs")
hubs_file = TABLES / "top_5_bottleneck_hubs.csv"
if hubs_file.exists():
    st.dataframe(pd.read_csv(hubs_file), width="stretch")

st.subheader("Top Delay Corridors")
corridors_file = TABLES / "top_delay_corridors.csv"
if corridors_file.exists():
    st.dataframe(pd.read_csv(corridors_file).head(20), width="stretch")

st.subheader("FTL vs Carting Recommendations")
policy_file = TABLES / "ftl_vs_carting_policy.csv"
if policy_file.exists():
    st.dataframe(pd.read_csv(policy_file).head(50), width="stretch")

if REPORT.exists():
    st.subheader("Strategy Memo")
    st.markdown(REPORT.read_text(encoding="utf-8"))

