import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, classification_report
import lightgbm as lgb
import shap

# === 1. Загружаем данные ===
df = pd.read_csv("C:/Users/Nuraly/Desktop/airport_etl/ml_part/flights_ml.csv", encoding="utf-16")



print("Данные:", df.shape)
print(df.head())

# === 2. Обработка ===
# Заполняем NaN нулями
df["tickets_sold_so_far"] = df["tickets_sold_so_far"].fillna(0)
df["avg_lead_time_days"] = df["avg_lead_time_days"].fillna(0)

# Фичи и таргет
features = [
    "dow", "month", "hour", "route_id",
    "great_circle_km", "aircraft_capacity",
    "tickets_sold_so_far", "avg_lead_time_days"
]
target = "is_dep_delay_15"

X = df[features]
y = df[target]

# === 3. TimeSeriesSplit ===
tscv = TimeSeriesSplit(n_splits=5)
logreg_results, lgb_results = [], []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # --- Logistic Regression ---
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    logreg_results.append((roc, pr, f1))

    print(f"\n=== Fold {fold} (LogReg) ===")
    print(f"ROC-AUC={roc:.3f}, PR-AUC={pr:.3f}, F1={f1:.3f}")
    print(classification_report(y_test, y_pred, digits=3))

    # --- LightGBM ---
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": "binary",
        "metric": ["auc", "average_precision"],
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.1
    }

    model = lgb.train(params, train_data, valid_sets=[test_data],
                      num_boost_round=200, callbacks=[lgb.early_stopping(20)])

    y_prob_lgb = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred_lgb = (y_prob_lgb > 0.5).astype(int)

    roc_lgb = roc_auc_score(y_test, y_prob_lgb)
    pr_lgb = average_precision_score(y_test, y_prob_lgb)
    f1_lgb = f1_score(y_test, y_pred_lgb)
    lgb_results.append((roc_lgb, pr_lgb, f1_lgb))

    print(f"\n=== Fold {fold} (LightGBM) ===")
    print(f"ROC-AUC={roc_lgb:.3f}, PR-AUC={pr_lgb:.3f}, F1={f1_lgb:.3f}")
    print(classification_report(y_test, y_pred_lgb, digits=3))

# === 4. Итог ===
logreg_avg = np.mean(logreg_results, axis=0)
lgb_avg = np.mean(lgb_results, axis=0)

print("\n>>> LogReg средние метрики:", dict(zip(["ROC-AUC","PR-AUC","F1"], logreg_avg)))
print(">>> LightGBM средние метрики:", dict(zip(["ROC-AUC","PR-AUC","F1"], lgb_avg)))

# === 5. Интерпретация признаков (SHAP) ===
print("\n>>> Интерпретация признаков (SHAP) для LightGBM...")
explainer = shap.Explainer(model, X.sample(2000, random_state=42))
shap_values = explainer(X.sample(2000, random_state=42))

shap.summary_plot(shap_values, X.sample(2000, random_state=42), feature_names=features)
