import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

df = pd.read_csv("flights_delay_ml.csv")
print("Размер:", df.shape)
print(df.head())

df = df.sort_values(["route_id", "flight_date", "hour"])
df["delay_rate_7d"] = (
    df.groupby(["route_id", "hour"])["is_dep_delay_15"]
      .transform(lambda x: x.shift().rolling(7, min_periods=1).mean())
)
df["delay_rate_30d"] = (
    df.groupby(["route_id", "hour"])["is_dep_delay_15"]
      .transform(lambda x: x.shift().rolling(30, min_periods=1).mean())
)
df = df.fillna(0)

features = [
    "hour", "dow", "month", "route_id", "great_circle_km",
    "aircraft_capacity", "tickets_sold_so_far", "avg_lead_time_days",
    "delay_rate_7d", "delay_rate_30d"
]
X = df[features]
y = df["is_dep_delay_15"]

tscv = TimeSeriesSplit(n_splits=5)
results = {"LogReg": [], "LightGBM": []}

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    logreg = LogisticRegression(max_iter=500)
    logreg.fit(X_train, y_train)
    pred_log = logreg.predict_proba(X_test)[:, 1]


    lgbm = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6)
    lgbm.fit(X_train, y_train)
    pred_lgb = lgbm.predict_proba(X_test)[:, 1]

    for name, pred in [("LogReg", pred_log), ("LightGBM", pred_lgb)]:
        roc = roc_auc_score(y_test, pred)
        pr = average_precision_score(y_test, pred)
        f1 = f1_score(y_test, (pred > 0.5).astype(int))
        print(f"Fold {fold} {name}: ROC={roc:.3f}, PR={pr:.3f}, F1={f1:.3f}")
        results[name].append((roc, pr, f1))

print("\n>>> Средние метрики по моделям:")
for name, scores in results.items():
    scores = np.array(scores)
    mean_roc, mean_pr, mean_f1 = scores.mean(axis=0)
    print(f"{name}: ROC={mean_roc:.3f}, PR={mean_pr:.3f}, F1={mean_f1:.3f}")

explainer = shap.TreeExplainer(lgbm)
sample = X.sample(2000, random_state=42)
shap_values = explainer.shap_values(sample)
shap.summary_plot(shap_values, sample, feature_names=features)
