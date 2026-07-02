# Payment Failure Forecasting

An ML pipeline and interactive dashboard that forecasts payment failure risk across
payment channels, routes, providers, and time windows — built for Optimus AI Labs
Hackathon 2026 (Test Case 2: Finance Domain).

🔗 **Live Dashboard:** [Add your Streamlit Cloud link here]

## Problem

Payment platforms (POS, transfers, USSD, cards, web checkout, agency banking) experience
failures caused by route instability, switch downtime, network issues, provider errors,
timeouts, and sudden transaction spikes. This project builds a system that predicts
*where* and *when* failure risk is rising — before it happens — so operations teams can
intervene proactively rather than reactively.

**Output format:** for each channel/route/provider/time-window combination, the system
returns a failure risk level (High/Medium/Low), the likely reason, and a recommended
operational action.

## Approach

The pipeline runs in four stages:

1. **EDA** — explores transaction volume, amount distributions, and time-of-day patterns
   in the base dataset
2. **Failure Injection** — since the base dataset has no real payment-failure signal,
   a synthetic failure-risk layer is engineered on top: routes, providers, a composite
   stress score, and a **probabilistic** (not deterministic) failure label — this
   avoids data leakage where the model could just reverse-engineer the labeling formula
3. **Feature Engineering** — five engineered features capture spikes, anomalies, and
   temporal/cross-route pressure (e.g. `volume_spike_ratio`, `rolling_failure_rate`,
   `provider_load_norm`)
4. **Modelling** — Random Forest and XGBoost are trained and compared, with SMOTE
   applied to the training set only to handle class imbalance

## Dataset

Base data: [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) — a synthetic
mobile-money transaction simulator calibrated to real transaction logs (Lopez-Rojas et
al., 2016). Chosen because its volume, timing, and amount distributions reflect authentic
mobile-money behaviour, providing a realistic foundation for the failure-risk layer
built on top.

## Results

| Metric | Random Forest | XGBoost |
|---|---|---|
| ROC-AUC (weighted OvR) | 0.868 | 0.869 |
| Accuracy | 80% | 80% |
| False alarm rate (High) | 3.96% | 3.83% |
| Missed incident rate (High) | 20.98% | 21.04% |

XGBoost was selected for the final dashboard due to a marginally lower false alarm rate
at equivalent discriminative power.

## Tech Stack

Python · pandas · scikit-learn · XGBoost · imbalanced-learn (SMOTE) · Streamlit ·
matplotlib/seaborn

## Repository Structure

- `main.py` — full pipeline: EDA, failure injection, feature engineering, model training
- `dashboard.py` — Streamlit app for interactive risk monitoring
- `dashboard_predictions.csv` — model output consumed by the dashboard
- `requirements.txt` — dependencies

## Responsible AI Note

This model is designed to support operational decision-making only. It must not be used
to automatically block transactions or deny customer service. All outputs are
recommendations for human review.

## Author

Oluwaseyi Fatai Abdulraman
[GitHub](https://github.com/oluwaseyifatai556)
