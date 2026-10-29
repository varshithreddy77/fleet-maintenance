"""
Fleet Predictive Maintenance - Scania APS failure model.

Predicts whether a truck's Air Pressure System (APS) is about to fail, using the
public "APS Failure at Scania Trucks" dataset (UCI ML Repository, GPL-3.0).

Business framing (from the dataset's own cost definition):
  - False positive  (inspect a healthy truck)      = cost 10
  - False negative  (miss a truck that then fails) = cost 500
Total cost = 10 * FP + 500 * FN. We pick the model and the decision threshold
that minimise this cost, then report it on the untouched test set.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from fleet_model import COST_FN, COST_FP, OutlierClipLog, add_missing_count, load_aps

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)

MISSING_DROP = 0.70  # drop sensors missing in more than 70% of trucks
SEED = 42




def total_cost(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return COST_FP * fp + COST_FN * fn, int(fp), int(fn), int(tp), int(tn)


def best_threshold(y, proba):
    grid = np.unique(np.round(np.r_[np.linspace(0.001, 0.2, 200), np.linspace(0.2, 0.9, 71)], 4))
    costs = [total_cost(y, (proba >= t).astype(int))[0] for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), int(costs[i])


def make_models():
    prep = [
        ("impute", SimpleImputer(strategy="median")),
        ("outliers", OutlierClipLog()),
    ]
    return {
        "logistic_regression": Pipeline(prep + [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", C=0.1, max_iter=3000)),
        ]),
        "decision_tree": Pipeline(prep + [
            ("model", DecisionTreeClassifier(class_weight="balanced", max_depth=8,
                                             min_samples_leaf=20, random_state=SEED)),
        ]),
        "random_forest": Pipeline(prep + [
            ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                             min_samples_leaf=2, max_features="sqrt",
                                             n_jobs=-1, random_state=SEED)),
        ]),
    }


def main():
    X_train, y_train = load_aps(DATA / "aps_failure_training_set.csv")
    X_test, y_test = load_aps(DATA / "aps_failure_test_set.csv")

    # --- Data quality: missing data + feature screening (fit on train only) ---
    miss = X_train.isna().mean()
    dropped = miss[miss > MISSING_DROP].index.tolist()
    keep = [c for c in X_train.columns if c not in dropped]
    features = keep + ["n_missing"]
    X_train = add_missing_count(X_train, keep)[features]
    X_test = add_missing_count(X_test, keep)[features]

    summary = {
        "train_rows": int(len(y_train)), "test_rows": int(len(y_test)),
        "raw_sensor_features": 170, "features_used": len(features),
        "sensors_dropped_high_missing": dropped,
        "train_failures": int(y_train.sum()), "test_failures": int(y_test.sum()),
        "pct_cells_missing_train": round(float(miss.mean() * 100), 1),
    }

    # --- Baselines on the test set ---
    base_cost_none = COST_FN * int(y_test.sum())                 # no model: every failure missed
    base_cost_all = COST_FP * int((y_test == 0).sum())           # inspect every truck
    summary["baseline_no_model_cost"] = base_cost_none
    summary["baseline_inspect_all_cost"] = base_cost_all

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}
    for name, pipe in make_models().items():
        # Threshold is tuned on out-of-fold training predictions, never on test.
        oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        thr, cv_cost = best_threshold(y_train, oof)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= thr).astype(int)
        cost, fp, fn, tp, tn = total_cost(y_test, pred)
        results[name] = {
            "threshold": thr, "cv_cost": cv_cost, "test_cost": int(cost),
            "fp": fp, "fn": fn, "tp": tp, "tn": tn,
            "recall": round(recall_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred), 4),
            "f1": round(f1_score(y_test, pred), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
            "cost_reduction_vs_no_model_pct": round((1 - cost / base_cost_none) * 100, 1),
            "cost_reduction_vs_inspect_all_pct": round((1 - cost / base_cost_all) * 100, 1),
        }
        print(name, json.dumps(results[name]))
        joblib.dump({"pipeline": pipe, "threshold": thr, "features": features, "sensor_cols": keep}, OUT / f"{name}.joblib")

    best = min(results, key=lambda k: results[k]["test_cost"])
    summary["models"] = results
    summary["best_model"] = best

    # Default-threshold (0.5) comparison for the best model, to show what tuning bought.
    bundle = joblib.load(OUT / f"{best}.joblib")
    p = bundle["pipeline"].predict_proba(X_test)[:, 1]
    c05 = total_cost(y_test, (p >= 0.5).astype(int))
    summary["best_model_default_threshold_cost"] = int(c05[0])

    # Top drivers for the app / write-up.
    model = bundle["pipeline"].named_steps["model"]
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        summary["top_features"] = imp.head(10).round(4).to_dict()

    joblib.dump(bundle, OUT / "best_model.joblib")
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("BEST:", best, json.dumps({k: v for k, v in summary.items() if k != "models"}, indent=2))


if __name__ == "__main__":
    main()
