from __future__ import annotations

import numpy as np
import pandas as pd
from node2vec import Node2Vec
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _within_15(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-6)
    return float(np.mean(np.abs(y_true - y_pred) / denom <= 0.15))


def train_baseline(df: pd.DataFrame) -> dict:
    # Baseline should be trip-level and non-graph.
    features = ["route_type", "time_bucket", "osrm_eta_minutes"]
    if "distance_km" in df.columns:
        features.append("distance_km")

    X = df[features]
    y = df["actual_transit_minutes"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cat_cols = ["route_type", "time_bucket"]
    num_cols = [c for c in features if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([("scale", StandardScaler())]), num_cols),
        ]
    )
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    return {
        "model": pipe,
        "mae": float(mean_absolute_error(y_test, preds)),
        "acc15": _within_15(y_test, preds),
        "test_df": X_test.copy(),
        "y_test": y_test,
        "preds": preds,
    }


def _node_embeddings(df: pd.DataFrame, dim: int = 16) -> dict[str, np.ndarray]:
    import networkx as nx

    g = nx.DiGraph()
    weighted = df.groupby(["source_facility", "dest_facility"]).size().reset_index(name="weight")
    for _, row in weighted.iterrows():
        g.add_edge(str(row["source_facility"]), str(row["dest_facility"]), weight=float(row["weight"]))

    if g.number_of_nodes() < 2:
        return {}

    n2v = Node2Vec(g, dimensions=dim, walk_length=20, num_walks=100, workers=1, weight_key="weight")
    w2v = n2v.fit(window=5, min_count=1)
    emb = {node: w2v.wv[node] for node in g.nodes()}
    return emb


def train_graph_enhanced(df: pd.DataFrame) -> dict:
    emb = _node_embeddings(df, dim=16)
    tmp = df.copy()
    for i in range(16):
        tmp[f"src_emb_{i}"] = tmp["source_facility"].astype(str).map(lambda x: emb.get(x, np.zeros(16))[i])
        tmp[f"dst_emb_{i}"] = tmp["dest_facility"].astype(str).map(lambda x: emb.get(x, np.zeros(16))[i])

    features = ["route_type", "time_bucket", "osrm_eta_minutes"] + [f"src_emb_{i}" for i in range(16)] + [
        f"dst_emb_{i}" for i in range(16)
    ]
    if "distance_km" in tmp.columns:
        features.append("distance_km")

    X = tmp[features]
    y = tmp["actual_transit_minutes"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cat_cols = ["route_type", "time_bucket"]
    num_cols = [c for c in features if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )
    model = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    return {
        "model": pipe,
        "mae": float(mean_absolute_error(y_test, preds)),
        "acc15": _within_15(y_test, preds),
    }

