"""Streamlit app: flag trucks at risk of APS failure and size the maintenance business case.

Run:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from fleet_model import COST_FN, COST_FP, OutlierClipLog, add_missing_count, load_aps  # noqa: F401

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
SAMPLE = ROOT / "data" / "aps_failure_test_set.csv"

st.set_page_config(page_title="Fleet Predictive Maintenance", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(ART / "best_model.joblib"), json.loads((ART / "metrics.json").read_text())


@st.cache_data
def read_upload(file):
    return load_aps(file)


bundle, metrics = load_bundle()
pipe, features, sensor_cols = bundle["pipeline"], bundle["features"], bundle["sensor_cols"]

st.title("Fleet Predictive Maintenance")
st.caption(
    "Random forest trained on 60,000 Scania trucks (APS failure dataset). "
    "Flags trucks likely to suffer an air-pressure-system failure and estimates "
    "whether a pre-emptive inspection pays for itself."
)

# ---------------- Inputs ----------------
with st.sidebar:
    st.header("Fleet data")
    up = st.file_uploader("Upload sensor CSV (Scania APS format)", type="csv")
    st.caption("No file? The app scores the 16,000-truck held-out test set.")
    st.header("Business assumptions")
    c_insp = st.number_input("Cost of one inspection", min_value=1.0, value=float(COST_FP), step=1.0)
    c_fail = st.number_input("Cost of one roadside failure", min_value=1.0, value=float(COST_FN), step=10.0)
    thr = st.slider("Flag threshold (risk score)", 0.0, 1.0, float(bundle["threshold"]), 0.001,
                    help=f"Default {bundle['threshold']} was tuned on cross-validated training data "
                         "to minimise total cost.")

X, y = read_upload(up) if up is not None else read_upload(SAMPLE)
missing = [c for c in sensor_cols if c not in X.columns]
if missing:
    st.error(f"Upload is missing {len(missing)} expected sensor columns, e.g. {missing[:5]}")
    st.stop()

Xf = add_missing_count(X, sensor_cols)[features]
risk = pipe.predict_proba(Xf)[:, 1]
flag = risk >= thr

# Per-truck economics: inspecting costs c_insp; not inspecting risks c_fail * risk.
expected_failure_cost = risk * c_fail
net_benefit = expected_failure_cost - c_insp
roi = net_benefit / c_insp

trucks = pd.DataFrame({
    "truck_id": np.arange(1, len(X) + 1),
    "risk_score": risk.round(4),
    "flagged": flag,
    "expected_failure_cost": expected_failure_cost.round(2),
    "inspection_net_benefit": net_benefit.round(2),
    "inspection_roi_x": roi.round(1),
})
if y is not None:
    trucks["actual_failure"] = y.astype(bool)

# ---------------- Fleet summary ----------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Trucks scored", f"{len(trucks):,}")
k2.metric("Flagged for inspection", f"{int(flag.sum()):,}", f"{flag.mean():.1%} of fleet")
k3.metric("Inspection spend", f"{flag.sum() * c_insp:,.0f}")
k4.metric("Expected failure cost avoided", f"{expected_failure_cost[flag].sum():,.0f}")

if y is not None:
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    tp = int((flag & (y == 1)).sum())
    model_cost = fp * c_insp + fn * c_fail
    no_model = int(y.sum()) * c_fail
    inspect_all = int((y == 0).sum()) * c_insp
    st.subheader("Business case (labelled data)")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Failures caught", f"{tp} of {int(y.sum())}", f"{tp / max(int(y.sum()), 1):.1%} recall")
    b2.metric("Total cost with model", f"{model_cost:,.0f}")
    b3.metric("Cost with no model", f"{no_model:,.0f}",
              f"-{1 - model_cost / no_model:.1%} with model", delta_color="inverse")
    b4.metric("Cost if every truck inspected", f"{inspect_all:,.0f}",
              f"-{1 - model_cost / inspect_all:.1%} with model", delta_color="inverse")
    st.caption(f"Model cost = {fp:,} unnecessary inspections x {c_insp:g} + {fn} missed failures x {c_fail:g}.")

# ---------------- Truck list ----------------
st.subheader("Trucks to inspect first")
st.dataframe(trucks[trucks.flagged].sort_values("risk_score", ascending=False),
             width="stretch", hide_index=True, height=360)
st.download_button("Download inspection list (CSV)",
                   trucks[trucks.flagged].to_csv(index=False), "inspection_list.csv", "text/csv")

# ---------------- Drivers ----------------
if "top_features" in metrics:
    st.subheader("Top failure drivers (model feature importance)")
    st.caption("Sensor names are anonymised by Scania; importance shows which readings drive risk most.")
    st.bar_chart(pd.Series(metrics["top_features"]).sort_values())

with st.expander("Model card"):
    m = metrics["models"][metrics["best_model"]]
    st.write({
        "model": metrics["best_model"],
        "decision_threshold": m["threshold"],
        "test_cost": m["test_cost"],
        "test_recall": m["recall"],
        "test_roc_auc": m["roc_auc"],
        "features_used": metrics["features_used"],
        "sensors_dropped_for_missingness": metrics["sensors_dropped_high_missing"],
    })
