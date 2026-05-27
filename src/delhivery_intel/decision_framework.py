from __future__ import annotations

import pandas as pd


def build_route_type_policy(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["source_facility", "dest_facility", "time_bucket", "route_type"]
    if "transport_cost" in df.columns:
        cost_expr = "mean"
    else:
        # If explicit cost is absent, use simple distance proxy.
        if "distance_km" in df.columns:
            df = df.copy()
            df["transport_cost"] = df["distance_km"] * df["route_type"].map({"FTL": 1.4, "Carting": 1.0}).fillna(1.1)
        else:
            df = df.copy()
            df["transport_cost"] = df["route_type"].map({"FTL": 140.0, "Carting": 100.0}).fillna(110.0)
        cost_expr = "mean"

    policy_base = (
        df.groupby(cols)
        .agg(
            avg_delay=("actual_transit_minutes", "mean"),
            breach_rate=("sla_breach", "mean"),
            transport_cost=("transport_cost", cost_expr),
        )
        .reset_index()
    )
    policy_base["expected_total_cost"] = policy_base["transport_cost"] + (policy_base["breach_rate"] * 500)

    pivot = policy_base.pivot_table(
        index=["source_facility", "dest_facility", "time_bucket"],
        columns="route_type",
        values="expected_total_cost",
        aggfunc="mean",
    ).reset_index()

    def pick(row: pd.Series) -> str:
        ftl = row.get("FTL")
        cart = row.get("Carting")
        if pd.isna(ftl) and pd.isna(cart):
            return "Insufficient Data"
        if pd.isna(ftl):
            return "Carting"
        if pd.isna(cart):
            return "FTL"
        return "FTL" if ftl < cart else "Carting"

    pivot["recommended_route_type"] = pivot.apply(pick, axis=1)
    return pivot

