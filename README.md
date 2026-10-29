# Fleet Predictive Maintenance

Predicts which trucks are about to suffer an Air Pressure System (APS) failure, and sizes the business case for inspecting them first.

Built on the public **APS Failure at Scania Trucks** dataset (UCI Machine Learning Repository, GPL-3.0): 60,000 training trucks and 16,000 held-out test trucks, 170 anonymised sensor readings each, with about 1 failure per 60 trucks.

## The business problem

Scania defines the costs directly:

| Decision | Cost |
|---|---|
| Inspect a healthy truck (false positive) | 10 |
| Miss a truck that then fails (false negative) | 500 |

Total cost = 10 x FP + 500 x FN. A model is only useful if it beats the two obvious policies: do nothing, or inspect every truck.

## Results (held-out test set, 16,000 trucks)

| Policy | Total cost | vs. no model |
|---|---|---|
| No model (every failure missed) | 187,500 | |
| Inspect every truck | 156,250 | -16.7% |
| Decision tree | 16,620 | -91.1% |
| Logistic regression | 13,510 | -92.8% |
| **Random forest (selected)** | **9,450** | **-95.0%** |

The random forest catches **365 of 375 failures (97.3% recall)** at the cost of 445 extra inspections.

Threshold tuning matters as much as the algorithm: the same random forest at the default 0.5 threshold costs **60,700**. Tuning the threshold to 0.059 on cross-validated training predictions cut that by 84%.

## Approach

1. **Missing data.** 8.3% of sensor cells are missing. Seven sensors missing in more than 70% of trucks are dropped; the rest are median-imputed, and a per-truck missing-sensor count is added as a feature.
2. **Outliers.** Sensors are heavy-tailed counters, so each is winsorised at its training 0.1/99.9 percentiles and log-transformed.
3. **Models.** Logistic regression, decision tree, and random forest, all class-weighted for the 59:1 imbalance.
4. **Cost-based threshold.** Each model's decision threshold is chosen to minimise total cost on 5-fold out-of-fold training predictions. The test set is never used for tuning.
5. **Application.** The selected model is embedded in a Streamlit app that scores a fleet, ranks trucks by risk, and estimates per-truck inspection ROI.

## Run it

```bash
pip install -r requirements.txt
python train.py          # trains, evaluates, writes artifacts/ (about 5 minutes)
streamlit run app.py     # opens the app
```

The trained model (`artifacts/best_model.joblib`) and the test set are included so the app runs straight away. To retrain, add `aps_failure_training_set.csv` from the UCI repository to `data/`.

## Files

- `fleet_model.py` – shared preprocessing (outlier clipping, missing-count feature, loader)
- `train.py` – training, threshold tuning, evaluation, `artifacts/metrics.json`
- `app.py` – Streamlit app
