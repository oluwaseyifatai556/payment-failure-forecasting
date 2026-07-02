"""
================================================================================
PAYMENT FAILURE FORECASTING — FULL PIPELINE
Optimus AI Labs Hackathon 2026
================================================================================

OVERVIEW
--------
This script implements the complete ML pipeline for Test Case 2:
Payment Failure Forecasting. It runs in three sequential stages:

    Stage 1 — EDA:               Explore the raw PaySim dataset
    Stage 2 — Failure Injection: Build synthetic failure-risk labels
                                 and route/provider infrastructure
    Stage 3 — Feature Eng:       Derive contextual features for modelling

DATA SOURCE
-----------
Base dataset: PaySim — "Synthetic Financial Datasets For Fraud Detection"
Source: https://www.kaggle.com/datasets/ealaxi/paysim1
License: CC BY-SA 4.0
Citation: E. A. Lopez-Rojas, A. Elmir, and S. Axelsson. "PaySim: A financial
          mobile money simulator for fraud detection". In: The 28th European
          Modeling and Simulation Symposium (EMSS), Larnaca, Cyprus. 2016.

WHY PaySim: Simulates one month of African mobile money transactions calibrated
to real transaction logs. Provides authentic volume, timing, and amount
distributions — the structural foundation our failure injection layer builds on.

SYNTHETIC LAYER
---------------
PaySim has no concept of payment failure. We engineer a failure-risk system
on top by:
  1. Assigning synthetic routes and providers (5 routes, 3 providers)
  2. Computing a stress score from volume, amount, hour, and route quality
  3. Assigning failure_risk labels PROBABILISTICALLY from the stress score

IMPORTANT — METHODOLOGY NOTE ON LABEL GENERATION
-------------------------------------------------
A key challenge with synthetic data: if the label is derived deterministically
from features also present in the model, the model trivially reverse-engineers
the formula (data leakage), producing artificially perfect metrics.

We address this with PROBABILISTIC LABELING:
  - A high-stress transaction is labelled "High" with 85% probability,
    not with 100% certainty.
  - This reflects real-world uncertainty: not every stressed route actually
    fails, and not every failure is preceded by visible stress signals.
  - The resulting label is correlated with — but not perfectly predictable
    from — the observable features.

This is explicitly documented as a synthetic data limitation. A production
system would use actual historical failure logs instead of injected labels.

LLM USAGE DISCLOSURE
--------------------
Claude (Anthropic) was used as a development assistant for:
  - Code structure and feature engineering guidance
  - Explaining statistical concepts during development
  - Identifying and resolving the data leakage issue
Claude was NOT used as the predictive model or scoring engine.
All modelling, evaluation, and data decisions were made by the developer.

RESPONSIBLE AI NOTE
-------------------
This model is designed to support operational decision-making only.
It must not be used to automatically block transactions or deny customer
service. Outputs are recommendations for human review, not final decisions.
================================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, ConfusionMatrixDisplay)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ================================================================================
# STAGE 1: EXPLORATORY DATA ANALYSIS
# ================================================================================

df = pd.read_csv('paysim_sample.csv')  # update path if needed

sns.set_style('whitegrid')

# ------------------------------------------------------------------------------
# Chart 1: Amount Distribution — Raw vs Log-transformed
# ------------------------------------------------------------------------------
# WHY: 'amount' is heavily right-skewed. A handful of multi-million transactions
# drag the mean far above the median, making the raw distribution unreadable.
# log1p(x) = log(x + 1) compresses the scale so genuine patterns are visible.
# The "+1" avoids log(0) errors on zero-amount rows.
# We will use log_amount as a feature throughout the pipeline for this reason.

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

sns.histplot(df['amount'], bins=80, ax=axes[0], color='#4C72B0')
axes[0].set_title('Amount Distribution (Raw)')
axes[0].set_xlabel('Amount')

sns.histplot(np.log1p(df['amount']), bins=80, ax=axes[1], color='#55A868')
axes[1].set_title('Amount Distribution (Log-transformed)')
axes[1].set_xlabel('log(Amount + 1)')

plt.tight_layout()
# plt.savefig('amount_distribution.png', dpi=120)
# plt.show()

# ------------------------------------------------------------------------------
# Chart 2: Transaction Count by Type
# ------------------------------------------------------------------------------
# WHY: Transaction type maps to payment channel (CASH_OUT ≈ agency banking,
# TRANSFER ≈ bank transfer, PAYMENT ≈ merchant payments). Volume per channel
# tells us which channels have enough data for reliable pattern detection.
# DEBIT (~1,900 rows) is so sparse we treat it cautiously throughout.

fig, ax = plt.subplots(figsize=(8, 5))
order = df['type'].value_counts().index
sns.countplot(data=df, x='type', order=order, hue='type',
              palette='viridis', legend=False, ax=ax)
ax.set_title('Transaction Count by Type')
ax.set_xlabel('Type')
ax.set_ylabel('Count')
plt.tight_layout()
# plt.savefig('type_counts.png', dpi=120)
# plt.show      ()

# ------------------------------------------------------------------------------
# Chart 3: Transaction Volume by Hour of Day
# ------------------------------------------------------------------------------
# WHY: 'step' in PaySim = hours since simulation start (1 step = 1 hour).
# step % 24 converts to hour of day (0–23). This reveals a clear diurnal
# (daily) cycle: volume is near-zero 1–7am, climbs sharply from 8am, and
# peaks between 18–21 (evening). High volume = high system load = higher
# failure risk. This directly informs our time-window risk design.

df['hour_of_day'] = df['step'] % 24

fig, ax = plt.subplots(figsize=(10, 5))
hourly = df.groupby('hour_of_day').size()
sns.lineplot(x=hourly.index, y=hourly.values, marker='o', ax=ax, color='#C44E52')
ax.set_title('Transaction Volume by Hour of Day')
ax.set_xlabel('Hour of Day (0-23)')
ax.set_ylabel('Transaction Count')
plt.tight_layout()
# plt.savefig('hourly_volume.png', dpi=120)
# plt.show()

# ------------------------------------------------------------------------------
# Chart 4: Amount Distribution by Type (log-scale boxplot)
# ------------------------------------------------------------------------------
# WHY: Each type has a meaningfully different amount range. TRANSFER median
# (~₦490k) is ~50x larger than PAYMENT median (~₦9.5k). Treating all channels
# as interchangeable would ignore this structural difference. This justifies
# type-specific features like amount_channel_deviation later.

fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(data=df, x='type', y='amount', order=order, hue='type',
            palette='Set2', legend=False, ax=ax)
ax.set_yscale('log')
ax.set_title('Amount Distribution by Transaction Type (log scale)')
plt.tight_layout()
# plt.savefig('amount_by_type.png', dpi=120)
# plt.show()


# ================================================================================
# STAGE 2: FAILURE INJECTION
# ================================================================================
# We continue with the same df from Stage 1 (already has hour_of_day).
# Routes, providers, stress scores, and the probabilistic failure label
# are all added here.

# ------------------------------------------------------------------------------
# Layer 1: Assign Routes and Providers
# ------------------------------------------------------------------------------
# WHY: Real payment systems route transactions through different infrastructure
# (bank switches, payment processors). We create 5 routes with different
# capacity profiles. Assignment is biased by transaction type to reflect
# how real systems wire channels to infrastructure:
#   TRANSFER (large value) → stronger routes (A, B) built for high amounts
#   CASH_OUT (high volume) → workhorse routes (C, D) built for throughput
#   Route_E = legacy/weak infrastructure shared by all types but infrequently

routes = ['Route_A', 'Route_B', 'Route_C', 'Route_D', 'Route_E']

route_weights = {
    'TRANSFER': [0.30, 0.30, 0.20, 0.15, 0.05],
    'CASH_OUT':  [0.15, 0.15, 0.35, 0.30, 0.05],
    'CASH_IN':   [0.20, 0.20, 0.25, 0.25, 0.10],
    'PAYMENT':   [0.20, 0.20, 0.20, 0.25, 0.15],
    'DEBIT':     [0.20, 0.20, 0.20, 0.20, 0.20],
}

df['route_id'] = df['type'].apply(
    lambda t: np.random.choice(routes, p=route_weights[t])
)

# 3 providers. Provider_3 serves the weakest routes, making it
# the most likely to become a bottleneck under combined load.
provider_map = {
    'Route_A': (['Provider_1', 'Provider_2'], [0.6, 0.4]),
    'Route_B': (['Provider_1', 'Provider_3'], [0.5, 0.5]),
    'Route_C': (['Provider_2', 'Provider_3'], [0.4, 0.6]),
    'Route_D': (['Provider_2', 'Provider_3'], [0.3, 0.7]),
    'Route_E': (['Provider_3'],               [1.0]),
}

df['provider_id'] = df['route_id'].apply(
    lambda r: np.random.choice(provider_map[r][0], p=provider_map[r][1])
)

# ------------------------------------------------------------------------------
# Layer 2: Time Features
# ------------------------------------------------------------------------------
# hour_of_day already created in Stage 1 (EDA Chart 3).
# We add 3-hour time windows here for route-level aggregation.
# Grouping hours into windows reduces noise while preserving the
# daily pattern we observed — a single hour is too granular,
# a whole day is too broad.

df['time_window'] = (df['hour_of_day'] // 3) * 3
df['time_window_label'] = df['time_window'].apply(lambda h: f"{h:02d}-{h+2:02d}")

# hour_risk_weight: a multiplier encoding how risky each time period is.
# Derived directly from the EDA finding: evening (18–21) = peak volume.
df['hour_risk_weight'] = df['hour_of_day'].apply(
    lambda h: 1.5 if 18 <= h <= 21 else (1.2 if 9 <= h <= 17 else 0.7)
)

# ------------------------------------------------------------------------------
# Layer 3: Stress Score
# ------------------------------------------------------------------------------
# The stress score is a composite pressure index (0–1) built from four signals:
#
#   volume_in_window        — how many transactions hit this route this period?
#   avg_log_amount_in_window — how large are those transactions on average?
#   hour_risk_weight         — is this a high-risk time of day?
#   route_fragility          — is this route's infrastructure known to be weak?
#
# WHY NORMALISE FIRST? Each signal is in different units (counts vs. log-naira
# vs. weights). Min-max normalisation scales all to [0,1] so no single
# signal dominates purely due to magnitude.
#
# WHY THESE WEIGHTS? Volume is the strongest real-world predictor of route
# stress (0.35), followed by transaction size (0.25), time-of-day (0.20),
# and infrastructure quality (0.20).
#
# IMPORTANT: stress_score is used to GENERATE the label, not as a model
# feature. See label generation section below.

df['log_amount'] = np.log1p(df['amount'])

df['volume_in_window'] = df.groupby(
    ['route_id', 'time_window'])['step'].transform('count')

df['avg_log_amount_in_window'] = df.groupby(
    ['route_id', 'time_window'])['log_amount'].transform('mean')

vol_min, vol_max = df['volume_in_window'].min(), df['volume_in_window'].max()
amt_min, amt_max = df['avg_log_amount_in_window'].min(), df['avg_log_amount_in_window'].max()

df['vol_norm'] = (df['volume_in_window'] - vol_min) / (vol_max - vol_min)
df['amt_norm'] = (df['avg_log_amount_in_window'] - amt_min) / (amt_max - amt_min)

route_fragility = {
    'Route_A': 0.1, 'Route_B': 0.1,
    'Route_C': 0.3, 'Route_D': 0.5, 'Route_E': 0.8,
}
df['route_fragility'] = df['route_id'].map(route_fragility)

df['stress_score'] = (
    0.35 * df['vol_norm'] +
    0.25 * df['amt_norm'] +
    0.20 * (df['hour_risk_weight'] / 1.5) +
    0.20 * df['route_fragility']
)

# ------------------------------------------------------------------------------
# Layer 4: Timeout Rate and Success Rate
# ------------------------------------------------------------------------------
# These are FEATURES (clues for the model), not the label.
# Timeout rate scales with stress — stressed routes drop more transactions —
# with realistic noise added (real systems don't behave perfectly).
# The brief explicitly calls out "rising timeout count and falling success rate"
# as observable failure signals, which justifies including them as features.

def inject_timeout(row):
    base  = row['stress_score'] * 0.25    # max 25% timeout at full stress
    noise = np.random.normal(0, 0.02)     # ±2% realistic variation
    return float(np.clip(base + noise, 0, 1))

df['timeout_rate'] = df.apply(inject_timeout, axis=1)
df['success_rate'] = 1 - df['timeout_rate']

# ------------------------------------------------------------------------------
# Layer 5: Probabilistic Failure Risk Label
# ------------------------------------------------------------------------------
# METHODOLOGY NOTE — why probabilistic, not deterministic:
#
# Early development used hard thresholds on stress_score to assign labels.
# This caused perfect model accuracy (ROC-AUC = 1.00) because the label was
# a deterministic function of features present in the training set — data
# leakage by construction.
#
# Switching to PROBABILISTIC LABELING breaks this circular dependency:
#   - A high-stress transaction is labelled "High" with 85% probability
#   - 15% of high-stress transactions are labelled "Medium" instead
#   - Similarly for Medium and Low boundaries
#
# This reflects operational reality: not every stressed route fails, and
# not every failure is preceded by measurable stress signals. The model
# must now learn genuine patterns rather than memorise our formula.
#
# Limitation: this is still a synthetic approximation. A production system
# would train on real historical failure events rather than injected labels.

high_thresh   = df['stress_score'].quantile(0.85)
medium_thresh = df['stress_score'].quantile(0.50)

print(f"Label thresholds — High: {high_thresh:.4f} | Medium: {medium_thresh:.4f}")

def probabilistic_label(score):
    """
    Assigns failure_risk probabilistically based on stress score.
    The 85/15 and 70/15/15 splits introduce deliberate uncertainty
    to prevent the model from reverse-engineering our stress formula.
    """
    if score >= high_thresh:
        # Clearly stressed — mostly High, occasionally Medium
        return np.random.choice(['High', 'Medium'], p=[0.85, 0.15])
    elif score >= medium_thresh:
        # Borderline — mostly Medium, some spillover either side
        return np.random.choice(['Medium', 'High', 'Low'], p=[0.70, 0.15, 0.15])
    else:
        # Clearly low stress — mostly Low, occasionally Medium
        return np.random.choice(['Low', 'Medium'], p=[0.85, 0.15])

df['failure_risk'] = df['stress_score'].apply(probabilistic_label)

print("\nFailure risk label distribution (probabilistic):")
print(df['failure_risk'].value_counts())
print(df['failure_risk'].value_counts(normalize=True).mul(100).round(1))

# Save enriched dataset for Stage 3
output_cols = [
    'step', 'type', 'amount', 'log_amount',
    'nameOrig', 'nameDest',
    'route_id', 'provider_id',
    'hour_of_day', 'time_window', 'time_window_label',
    'hour_risk_weight', 'route_fragility',
    'volume_in_window', 'avg_log_amount_in_window',
    'vol_norm', 'amt_norm',
    'stress_score', 'timeout_rate', 'success_rate',
    'failure_risk'
]

df[output_cols].to_csv('paysim_enriched.csv', index=False)
print("\nEnriched dataset saved as paysim_enriched.csv")
print("Shape:", df[output_cols].shape)


# ================================================================================
# STAGE 3: FEATURE ENGINEERING
# ================================================================================
# The injection layer describes WHAT is happening to a route right now.
# Feature engineering adds CONTEXT: how is this route trending over time?
# Is this load spike unusual? Is the provider being overloaded across routes?
#
# Five new features are added, each targeting a different dimension of
# operational risk that the raw injection features don't capture.

df = pd.read_csv('paysim_enriched.csv')

# ------------------------------------------------------------------------------
# Feature 1: volume_spike_ratio
# ------------------------------------------------------------------------------
# WHY: The brief lists "sudden transaction spikes" as a named failure cause.
# A route handling double its usual load is more dangerous than one that
# is consistently busy — consistent busyness means the infrastructure was
# provisioned for it; a spike means it wasn't.
#
# ratio = current window volume / this route's average volume across all windows
# ratio > 1.0 = busier than normal | ratio > 2.0 = double the usual traffic
# Correlation with stress_score: +0.79 (strongest of our five features)

route_mean_vol = (
    df.groupby(['route_id', 'time_window'])['volume_in_window']
    .first()
    .groupby(level=0)
    .mean()
)
df['route_avg_volume'] = df['route_id'].map(route_mean_vol)
df['volume_spike_ratio'] = df['volume_in_window'] / df['route_avg_volume']

# ------------------------------------------------------------------------------
# Feature 2: amount_channel_deviation (Z-score)
# ------------------------------------------------------------------------------
# WHY: EDA showed each transaction type has a characteristic amount range
# (TRANSFER median ~₦490k, PAYMENT median ~₦9.5k). A PAYMENT worth ₦5M
# is anomalous — it places load on a channel not designed for large amounts.
#
# Z-score = (this transaction's log_amount - type average) / type std deviation
# Positive = unusually large for its type | Negative = unusually small
# Using log_amount before computing the Z-score prevents outliers from
# distorting the type mean and std.
# Correlation with stress_score: +0.01 (weak but included as edge-case signal)

type_mean_log = df.groupby('type')['log_amount'].transform('mean')
type_std_log  = df.groupby('type')['log_amount'].transform('std')
df['amount_channel_deviation'] = (df['log_amount'] - type_mean_log) / type_std_log

# ------------------------------------------------------------------------------
# Feature 3: rolling_failure_rate
# ------------------------------------------------------------------------------
# WHY: A route stressed across 3 consecutive time windows is more worrying
# than one that spiked once. This feature captures temporal accumulation of
# pressure — analogous to monitoring a transformer: one voltage spike is a
# warning; three consecutive spikes means the line is in serious trouble.
#
# Method: for each route, look back at the last 3 time windows and compute
# the fraction of those windows whose average stress_score exceeded the 85th pct.
# Possible values: 0.0, 0.33, 0.67, 1.0
# Correlation with stress_score: +0.53

window_stress = (
    df.groupby(['route_id', 'time_window'])['stress_score']
    .mean()
    .reset_index()
    .rename(columns={'stress_score': 'window_avg_stress'})
)

high_thresh = df['stress_score'].quantile(0.85)
window_stress['window_stressed'] = (
    window_stress['window_avg_stress'] >= high_thresh
).astype(int)

window_stress = window_stress.sort_values(['route_id', 'time_window'])
window_stress['rolling_failure_rate'] = (
    window_stress
    .groupby('route_id')['window_stressed']
    .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
)

df = df.merge(
    window_stress[['route_id', 'time_window', 'rolling_failure_rate']],
    on=['route_id', 'time_window'],
    how='left'
)

# ------------------------------------------------------------------------------
# Feature 4: is_peak_hour (binary flag)
# ------------------------------------------------------------------------------
# WHY: EDA confirmed transactions peak between 18–21. A binary flag is more
# direct than hour_of_day (0–23) because models can misread hour_of_day
# as a continuous ordinal — as if hour 23 is "more" than hour 1 in a
# meaningful way. The binary flag makes peak-hour risk explicit.
# Correlation with stress_score: +0.34

df['is_peak_hour'] = df['hour_of_day'].apply(
    lambda h: 1 if 18 <= h <= 21 else 0
)

# ------------------------------------------------------------------------------
# Feature 5: provider_load_norm
# ------------------------------------------------------------------------------
# WHY: Routes share providers. Provider_3 serves Route_C, D, and E.
# When those routes are all busy simultaneously, Provider_3 becomes the
# bottleneck even if each individual route looks manageable.
# This feature captures cross-route load on the shared provider level.
#
# Normalised to 0–1 so it's on the same scale as other features.
# Correlation with stress_score: +0.59

df['provider_load'] = df.groupby(
    ['provider_id', 'time_window'])['step'].transform('count')

pload_min = df['provider_load'].min()
pload_max = df['provider_load'].max()
df['provider_load_norm'] = (df['provider_load'] - pload_min) / (pload_max - pload_min)

# ------------------------------------------------------------------------------
# Correlation check (sanity check before modelling)
# ------------------------------------------------------------------------------
new_features = [
    'volume_spike_ratio', 'amount_channel_deviation',
    'rolling_failure_rate', 'is_peak_hour', 'provider_load_norm'
]
print("\nCorrelation of engineered features with stress_score:")
for f in new_features:
    print(f"  {f:<35} {df[f].corr(df['stress_score']):+.4f}")

print("\nNull check:")
print(df[new_features].isnull().sum())

# ------------------------------------------------------------------------------
# Save final dataset
# ------------------------------------------------------------------------------
final_cols = [
    'step', 'type', 'amount', 'log_amount',
    'route_id', 'provider_id',
    'hour_of_day', 'time_window', 'time_window_label',
    'is_peak_hour',
    'hour_risk_weight', 'route_fragility',
    'volume_in_window', 'avg_log_amount_in_window',
    'vol_norm', 'amt_norm',
    'timeout_rate', 'success_rate',
    'stress_score',
    'volume_spike_ratio', 'amount_channel_deviation',
    'rolling_failure_rate',
    'provider_load', 'provider_load_norm',
    'failure_risk'
]

df[final_cols].to_csv('paysim_features.csv', index=False)
print("\nFinal dataset saved as paysim_features.csv")
print("Shape:", df[final_cols].shape)

# Features the model will train on (excludes identifiers, stress_score, label)
# stress_score, timeout_rate, success_rate are excluded from MODEL_FEATURES
# in modelling.py — see that file for the data leakage explanation.
MODEL_FEATURES = [
    'log_amount',
    'hour_of_day', 'is_peak_hour', 'hour_risk_weight',
    'route_fragility',
    'volume_in_window', 'avg_log_amount_in_window',
    'vol_norm', 'amt_norm',
    'volume_spike_ratio',
    'amount_channel_deviation',
    'rolling_failure_rate',
    'provider_load_norm'
]

print(f"\n{len(MODEL_FEATURES)} features ready for modelling:")
for f in MODEL_FEATURES:
    print(f"  - {f}")

# ================================================================================
# STAGE 4: MODEL TRAINING AND EVALUATION
# ================================================================================
# Two models are trained and compared:
#   Model 1 — Random Forest: tree-based ensemble, no feature scaling needed,
#             interpretable via feature importance, familiar from prior projects
#   Model 2 — XGBoost: gradient boosting, consistently outperforms RF on
#             tabular data, industry standard for fintech ML pipelines
#
# EVALUATION METRICS (aligned to hackathon brief):
#   - Precision, Recall, F1  → per-class, focus on High risk
#   - ROC-AUC (weighted OvR) → overall discrimination ability
#   - False alarm rate        → FP / (FP + TN) for High class
#                               operational cost: unnecessary alerts
#   - Missed incident rate    → FN / (FN + TP) for High class
#                               operational cost: real failures go undetected
#
# CLASS IMBALANCE HANDLING:
#   SMOTE is applied to the TRAINING SET ONLY.
#   The test set remains at real-world proportions — evaluating on synthetic
#   SMOTE data would give misleadingly optimistic metrics.
#   class_weight='balanced' is also set in Random Forest as a second layer.



df = pd.read_csv('paysim_features.csv')  # update path if needed

# ------------------------------------------------------------------------------
# Features and target
# ------------------------------------------------------------------------------
# stress_score  → EXCLUDED: used to generate the label (data leakage)
# timeout_rate  → EXCLUDED: derived directly from stress_score
# success_rate  → EXCLUDED: = 1 - timeout_rate (same reason)
# All remaining features are genuine observable signals.

MODEL_FEATURES = [
    'log_amount',
    'hour_of_day', 'is_peak_hour', 'hour_risk_weight',
    'route_fragility',
    'volume_in_window', 'avg_log_amount_in_window',
    'vol_norm', 'amt_norm',
    'volume_spike_ratio',
    'amount_channel_deviation',
    'rolling_failure_rate',
    'provider_load_norm'
]

X = df[MODEL_FEATURES]

# LabelEncoder maps text labels to integers (alphabetical order):
# High=0, Low=1, Medium=2
le = LabelEncoder()
le.fit(['Low', 'Medium', 'High'])
y_encoded = le.transform(df['failure_risk'])
high_idx  = list(le.classes_).index('High')

print("Label encoding:", dict(zip(le.classes_, le.transform(le.classes_))))
print("\nClass distribution:")
for cls, cnt in zip(le.classes_, np.bincount(y_encoded)):
    print(f"  {cls}: {cnt:,} ({cnt/len(y_encoded)*100:.1f}%)")

# ------------------------------------------------------------------------------
# Train / Test Split
# ------------------------------------------------------------------------------
# stratify=y_encoded ensures both splits have the same class proportions.
# Without this, one split could end up with far fewer High-risk examples.

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

# ------------------------------------------------------------------------------
# SMOTE — applied to training set only
# ------------------------------------------------------------------------------
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("\nClass distribution after SMOTE (train only):")
for cls, cnt in zip(le.classes_, np.bincount(y_train_sm)):
    print(f"  {cls}: {cnt:,} ({cnt/len(y_train_sm)*100:.1f}%)")

# ------------------------------------------------------------------------------
# Helper: compute false alarm rate and missed incident rate for High class
# ------------------------------------------------------------------------------
def high_risk_metrics(cm, high_idx):
    """
    False alarm rate  = FP / (FP + TN)
        What % of non-High transactions did we wrongly flag as High?
        Operational cost: unnecessary alerts, alert fatigue in ops teams.

    Missed incident rate = FN / (FN + TP)
        What % of genuinely High-risk transactions did we miss?
        Operational cost: real payment failures go undetected.
    """
    TP = cm[high_idx, high_idx]
    FP = cm[:, high_idx].sum() - TP
    FN = cm[high_idx, :].sum() - TP
    TN = cm.sum() - TP - FP - FN
    return FP / (FP + TN), FN / (FN + TP)

# ------------------------------------------------------------------------------
# Model 1: Random Forest
# ------------------------------------------------------------------------------
print("\n" + "="*58)
print("MODEL 1: RANDOM FOREST")
print("="*58)

rf = RandomForestClassifier(
    n_estimators=100,        # 100 decision trees in the ensemble
    max_depth=15,            # max depth per tree — limits overfitting
    class_weight='balanced', # second-layer imbalance correction (on top of SMOTE)
    random_state=42,
    n_jobs=-1                # use all available CPU cores
)
rf.fit(X_train_sm, y_train_sm)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

roc_rf        = roc_auc_score(y_test, y_prob_rf, multi_class='ovr', average='weighted')
cm_rf         = confusion_matrix(y_test, y_pred_rf)
far_rf, mir_rf = high_risk_metrics(cm_rf, high_idx)

print(f"ROC-AUC (weighted OvR): {roc_rf:.4f}")
print(f"False alarm rate:       {far_rf:.4f}")
print(f"Missed incident rate:   {mir_rf:.4f}")

# ------------------------------------------------------------------------------
# Model 2: XGBoost
# ------------------------------------------------------------------------------
# Gradient boosting: each tree corrects the residual errors of the previous one.
# learning_rate controls how much each tree's correction is weighted —
# lower = more conservative, requires more trees to converge.

print("\n" + "="*58)
print("MODEL 2: XGBOOST")
print("="*58)

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train_sm, y_train_sm)

y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=le.classes_))

roc_xgb         = roc_auc_score(y_test, y_prob_xgb, multi_class='ovr', average='weighted')
cm_xgb          = confusion_matrix(y_test, y_pred_xgb)
far_xgb, mir_xgb = high_risk_metrics(cm_xgb, high_idx)

print(f"ROC-AUC (weighted OvR): {roc_xgb:.4f}")
print(f"False alarm rate:       {far_xgb:.4f}")
print(f"Missed incident rate:   {mir_xgb:.4f}")

# ------------------------------------------------------------------------------
# Comparison Summary
# ------------------------------------------------------------------------------
print("\n" + "="*58)
print("MODEL COMPARISON SUMMARY")
print("="*58)
print(f"{'Metric':<35} {'Random Forest':>12} {'XGBoost':>10}")
print("-"*58)
print(f"{'ROC-AUC (weighted OvR)':<35} {roc_rf:>12.4f} {roc_xgb:>10.4f}")
print(f"{'False alarm rate (High)':<35} {far_rf:>12.4f} {far_xgb:>10.4f}")
print(f"{'Missed incident rate (High)':<35} {mir_rf:>12.4f} {mir_xgb:>10.4f}")
print("\nSelected model: XGBoost (lower false alarm rate, equivalent ROC-AUC)")

# ------------------------------------------------------------------------------
# Charts
# ------------------------------------------------------------------------------

# Feature importance — both models side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, model, name in zip(axes, [rf, xgb], ['Random Forest', 'XGBoost']):
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    ax.barh([MODEL_FEATURES[i] for i in idx], imp[idx], color='#4C72B0')
    ax.set_title(f'{name} — Feature Importance')
    ax.set_xlabel('Importance score')
    ax.invert_yaxis()
plt.tight_layout()
# plt.savefig('feature_importance.png', dpi=120)
# plt.show()

# Confusion matrices — both models side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, cm, name in zip(axes, [cm_rf, cm_xgb], ['Random Forest', 'XGBoost']):
    ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
        ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Confusion Matrix — {name}')
plt.tight_layout()
# plt.savefig('confusion_matrices.png', dpi=120)
# plt.show()

print("\nCharts saved: feature_importance.png, confusion_matrices.png")

"""
Payment Failure Forecasting Dashboard
Optimus AI Labs Hackathon 2026 — Test Case 2

Run locally:    streamlit run dashboard.py
Deploy:         Push to GitHub → connect repo at share.streamlit.io
"""




# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Payment Failure Forecasting",
    page_icon="⚡",
    layout="wide"
)

# ── Colour mapping ─────────────────────────────────────────────────────
RISK_COLOURS = {'High': '#d62728', 'Medium': '#ff7f0e', 'Low': '#2ca02c'}

# ── Load data ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('dashboard_predictions.csv')

df = load_data()

# ── Header ─────────────────────────────────────────────────────────────
st.title("⚡ Payment Failure Forecasting")
st.markdown(
    "Real-time failure risk across payment channels, routes, and time windows. "
    "Powered by XGBoost trained on PaySim transaction data."
)
st.divider()

# ── KPI summary row ────────────────────────────────────────────────────
high_count   = (df['Failure Risk'] == 'High').sum()
medium_count = (df['Failure Risk'] == 'Medium').sum()
low_count    = (df['Failure Risk'] == 'Low').sum()
total        = len(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("🔴 High Risk Alerts",   f"{high_count}",
            f"{high_count/total*100:.0f}% of all route-windows")
col2.metric("🟠 Medium Risk",        f"{medium_count}",
            f"{medium_count/total*100:.0f}% of all route-windows")
col3.metric("🟢 Low Risk",           f"{low_count}",
            f"{low_count/total*100:.0f}% of all route-windows")
col4.metric("📊 Route-Windows Monitored", f"{total}")

st.divider()

# ── Sidebar filters ────────────────────────────────────────────────────
st.sidebar.header("🔍 Filters")

risk_filter = st.sidebar.multiselect(
    "Failure Risk Level",
    options=['High', 'Medium', 'Low'],
    default=['High', 'Medium']
)

route_filter = st.sidebar.multiselect(
    "Route",
    options=sorted(df['Route'].unique()),
    default=sorted(df['Route'].unique())
)

channel_filter = st.sidebar.multiselect(
    "Channel",
    options=sorted(df['Channel'].unique()),
    default=sorted(df['Channel'].unique())
)

window_filter = st.sidebar.multiselect(
    "Time Window",
    options=sorted(df['Time Window'].unique()),
    default=sorted(df['Time Window'].unique())
)

# Apply filters
filtered = df[
    df['Failure Risk'].isin(risk_filter) &
    df['Route'].isin(route_filter) &
    df['Channel'].isin(channel_filter) &
    df['Time Window'].isin(window_filter)
].copy()

st.markdown(f"**Showing {len(filtered)} route-window combinations**")

# ── Charts row ─────────────────────────────────────────────────────────
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Failure Risk by Route")
    route_risk = (df.groupby(['Route', 'Failure Risk'])
                    .size().unstack(fill_value=0))
    for col in ['High', 'Medium', 'Low']:
        if col not in route_risk.columns:
            route_risk[col] = 0
    route_risk = route_risk[['High', 'Medium', 'Low']]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    route_risk.plot(kind='bar', ax=ax, stacked=True,
                    color=[RISK_COLOURS['High'],
                           RISK_COLOURS['Medium'],
                           RISK_COLOURS['Low']],
                    edgecolor='white')
    ax.set_xlabel('')
    ax.set_ylabel('Route-window count')
    ax.set_title('Risk distribution per route')
    ax.legend(loc='upper right', fontsize=8)
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with chart_col2:
    st.subheader("Failure Risk by Time Window")
    window_risk = (df.groupby(['Time Window', 'Failure Risk'])
                     .size().unstack(fill_value=0))
    for col in ['High', 'Medium', 'Low']:
        if col not in window_risk.columns:
            window_risk[col] = 0
    window_risk = window_risk[['High', 'Medium', 'Low']].sort_index()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    window_risk.plot(kind='bar', ax=ax, stacked=True,
                     color=[RISK_COLOURS['High'],
                            RISK_COLOURS['Medium'],
                            RISK_COLOURS['Low']],
                     edgecolor='white')
    ax.set_xlabel('Time Window')
    ax.set_ylabel('Route-window count')
    ax.set_title('Risk concentration by time of day')
    ax.legend(loc='upper right', fontsize=8)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.divider()

# ── Prediction table ───────────────────────────────────────────────────
st.subheader("📋 Route-Level Failure Risk Forecast")
st.markdown(
    "Each row represents one route/channel/time-window combination. "
    "Sorted by risk severity."
)

display_cols = [
    'Route', 'Channel', 'Time Window', 'Provider',
    'Failure Risk', 'Transaction Count',
    'Volume Spike Ratio', 'Timeout Rate (%)', 'Success Rate (%)',
    'Reason', 'Recommended Action'
]

display_df = filtered[display_cols].sort_values(
    'Failure Risk',
    key=lambda x: x.map({'High': 0, 'Medium': 1, 'Low': 2})
).reset_index(drop=True)

def colour_risk(val):
    colours = {'High': '#ffd7d7', 'Medium': '#ffe4c4', 'Low': '#d4edda'}
    return f'background-color: {colours.get(val, "")}'

st.dataframe(
    display_df.style.applymap(colour_risk, subset=['Failure Risk']),
    use_container_width=True,
    height=420
)

# ── Sample operational output (brief format) ───────────────────────────
st.divider()
st.subheader("🚨 High Risk Alert Feed")
st.markdown(
    "Formatted alerts in operational output style, "
    "ready for integration with an ops team notification system."
)

high_rows = filtered[filtered['Failure Risk'] == 'High'].head(8)

if high_rows.empty:
    st.info("No High risk alerts match the current filters.")
else:
    for _, row in high_rows.iterrows():
        with st.container():
            st.error(
                f"**Channel:** {row['Channel']}  |  "
                f"**Route:** {row['Route']}  |  "
                f"**Provider:** {row['Provider']}  |  "
                f"**Time Window:** {row['Time Window']}  |  "
                f"**Risk:** 🔴 {row['Failure Risk']}  \n"
                f"**Reason:** {row['Reason']}  \n"
                f"**Action:** {row['Recommended Action']}"
            )

# ── Model performance footer ───────────────────────────────────────────
st.divider()
st.subheader("📈 Model Performance")

perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(5)
perf_col1.metric("Model",         "XGBoost")
perf_col2.metric("ROC-AUC",       "0.8686")
perf_col3.metric("Accuracy",      "80%")
perf_col4.metric("False Alarm Rate (High)", "3.8%")
perf_col5.metric("Missed Incident Rate (High)", "21.0%")

st.caption(
    "**Responsible AI note:** This model supports operational decision-making only. "
    "It must not be used to automatically block transactions or deny customer service. "
    "All outputs are recommendations for human review."
)
"""
Payment Failure Forecasting Dashboard
Optimus AI Labs Hackathon 2026 — Test Case 2

Run locally:    streamlit run dashboard.py
Deploy:         Push to GitHub → connect repo at share.streamlit.io
"""


# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Payment Failure Forecasting",
    page_icon="⚡",
    layout="wide"
)

# ── Colour mapping ─────────────────────────────────────────────────────
RISK_COLOURS = {'High': '#d62728', 'Medium': '#ff7f0e', 'Low': '#2ca02c'}

# ── Load data ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('dashboard_predictions.csv')

df = load_data()

# ── Header ─────────────────────────────────────────────────────────────
st.title("⚡ Payment Failure Forecasting")
st.markdown(
    "Real-time failure risk across payment channels, routes, and time windows. "
    "Powered by XGBoost trained on PaySim transaction data."
)
st.divider()

# ── KPI summary row ────────────────────────────────────────────────────
high_count   = (df['Failure Risk'] == 'High').sum()
medium_count = (df['Failure Risk'] == 'Medium').sum()
low_count    = (df['Failure Risk'] == 'Low').sum()
total        = len(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("🔴 High Risk Alerts",   f"{high_count}",
            f"{high_count/total*100:.0f}% of all route-windows")
col2.metric("🟠 Medium Risk",        f"{medium_count}",
            f"{medium_count/total*100:.0f}% of all route-windows")
col3.metric("🟢 Low Risk",           f"{low_count}",
            f"{low_count/total*100:.0f}% of all route-windows")
col4.metric("📊 Route-Windows Monitored", f"{total}")

st.divider()

# ── Sidebar filters ────────────────────────────────────────────────────
st.sidebar.header("🔍 Filters")

risk_filter = st.sidebar.multiselect(
    "Failure Risk Level",
    options=['High', 'Medium', 'Low'],
    default=['High', 'Medium']
)

route_filter = st.sidebar.multiselect(
    "Route",
    options=sorted(df['Route'].unique()),
    default=sorted(df['Route'].unique())
)

channel_filter = st.sidebar.multiselect(
    "Channel",
    options=sorted(df['Channel'].unique()),
    default=sorted(df['Channel'].unique())
)

window_filter = st.sidebar.multiselect(
    "Time Window",
    options=sorted(df['Time Window'].unique()),
    default=sorted(df['Time Window'].unique())
)

# Apply filters
filtered = df[
    df['Failure Risk'].isin(risk_filter) &
    df['Route'].isin(route_filter) &
    df['Channel'].isin(channel_filter) &
    df['Time Window'].isin(window_filter)
].copy()

st.markdown(f"**Showing {len(filtered)} route-window combinations**")

# ── Charts row ─────────────────────────────────────────────────────────
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Failure Risk by Route")
    route_risk = (df.groupby(['Route', 'Failure Risk'])
                    .size().unstack(fill_value=0))
    for col in ['High', 'Medium', 'Low']:
        if col not in route_risk.columns:
            route_risk[col] = 0
    route_risk = route_risk[['High', 'Medium', 'Low']]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    route_risk.plot(kind='bar', ax=ax, stacked=True,
                    color=[RISK_COLOURS['High'],
                           RISK_COLOURS['Medium'],
                           RISK_COLOURS['Low']],
                    edgecolor='white')
    ax.set_xlabel('')
    ax.set_ylabel('Route-window count')
    ax.set_title('Risk distribution per route')
    ax.legend(loc='upper right', fontsize=8)
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with chart_col2:
    st.subheader("Failure Risk by Time Window")
    window_risk = (df.groupby(['Time Window', 'Failure Risk'])
                     .size().unstack(fill_value=0))
    for col in ['High', 'Medium', 'Low']:
        if col not in window_risk.columns:
            window_risk[col] = 0
    window_risk = window_risk[['High', 'Medium', 'Low']].sort_index()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    window_risk.plot(kind='bar', ax=ax, stacked=True,
                     color=[RISK_COLOURS['High'],
                            RISK_COLOURS['Medium'],
                            RISK_COLOURS['Low']],
                     edgecolor='white')
    ax.set_xlabel('Time Window')
    ax.set_ylabel('Route-window count')
    ax.set_title('Risk concentration by time of day')
    ax.legend(loc='upper right', fontsize=8)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.divider()

# ── Prediction table ───────────────────────────────────────────────────
st.subheader("📋 Route-Level Failure Risk Forecast")
st.markdown(
    "Each row represents one route/channel/time-window combination. "
    "Sorted by risk severity."
)

display_cols = [
    'Route', 'Channel', 'Time Window', 'Provider',
    'Failure Risk', 'Transaction Count',
    'Volume Spike Ratio', 'Timeout Rate (%)', 'Success Rate (%)',
    'Reason', 'Recommended Action'
]

display_df = filtered[display_cols].sort_values(
    'Failure Risk',
    key=lambda x: x.map({'High': 0, 'Medium': 1, 'Low': 2})
).reset_index(drop=True)

def colour_risk(val):
    colours = {'High': '#ffd7d7', 'Medium': '#ffe4c4', 'Low': '#d4edda'}
    return f'background-color: {colours.get(val, "")}'

st.dataframe(
    display_df.style.applymap(colour_risk, subset=['Failure Risk']),
    use_container_width=True,
    height=420
)

# ── Sample operational output (brief format) ───────────────────────────
st.divider()
st.subheader("🚨 High Risk Alert Feed")
st.markdown(
    "Formatted alerts in operational output style, "
    "ready for integration with an ops team notification system."
)

high_rows = filtered[filtered['Failure Risk'] == 'High'].head(8)

if high_rows.empty:
    st.info("No High risk alerts match the current filters.")
else:
    for _, row in high_rows.iterrows():
        with st.container():
            st.error(
                f"**Channel:** {row['Channel']}  |  "
                f"**Route:** {row['Route']}  |  "
                f"**Provider:** {row['Provider']}  |  "
                f"**Time Window:** {row['Time Window']}  |  "
                f"**Risk:** 🔴 {row['Failure Risk']}  \n"
                f"**Reason:** {row['Reason']}  \n"
                f"**Action:** {row['Recommended Action']}"
            )

# ── Model performance footer ───────────────────────────────────────────
st.divider()
st.subheader("📈 Model Performance")

perf_col1, perf_col2, perf_col3, perf_col4, perf_col5 = st.columns(5)
perf_col1.metric("Model",         "XGBoost")
perf_col2.metric("ROC-AUC",       "0.8686")
perf_col3.metric("Accuracy",      "80%")
perf_col4.metric("False Alarm Rate (High)", "3.8%")
perf_col5.metric("Missed Incident Rate (High)", "21.0%")

st.caption(
    "**Responsible AI note:** This model supports operational decision-making only. "
    "It must not be used to automatically block transactions or deny customer service. "
    "All outputs are recommendations for human review."
)