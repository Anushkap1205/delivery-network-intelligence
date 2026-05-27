from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_top_hubs(top_hubs: pd.DataFrame, out_path: str) -> None:
    if top_hubs.empty:
        return
    plt.figure(figsize=(8, 4))
    df = top_hubs.sort_values("hub_risk_score", ascending=True)
    plt.barh(df["hub"], df["hub_risk_score"])
    plt.title("Top Bottleneck Hubs by Risk Score")
    plt.xlabel("Hub Risk Score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_delay_corridors(corridors: pd.DataFrame, out_path: str) -> None:
    if corridors.empty:
        return
    plt.figure(figsize=(10, 4))
    df = corridors.head(15).copy()
    names = df["source_facility"].astype(str) + "->" + df["dest_facility"].astype(str)
    plt.bar(range(len(df)), df["avg_delay_pct"] * 100)
    plt.xticks(range(len(df)), names, rotation=60, ha="right")
    plt.ylabel("Avg Delay %")
    plt.title("Chronic Delay Corridors")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

