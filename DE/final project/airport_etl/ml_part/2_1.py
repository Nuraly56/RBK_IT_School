import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# === 1. Загрузка данных ===
df = pd.read_csv("C:/Users/Nuraly/Desktop/airport_etl/ml_part/flights_for_demand.csv")

print("Данные:", df.shape)
print(df.head())

# === 2. Предобработка ===
df["tickets_sold"] = df["tickets_sold"].fillna(0)
df["avg_lead_time_days"] = df["avg_lead_time_days"].fillna(0)

# Фичи и таргет
X = df[["route_id", "distance_km", "aircraft_capacity",
        "avg_lead_time_days", "dow", "month", "year"]]
y = df["tickets_sold"]

print("Размер X:", X.shape, "Размер y:", y.shape)

# === 3. Временная кросс-валидация ===
tscv = TimeSeriesSplit(n_splits=5)

rmse_scores, smape_scores = [], []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # LightGBM модель
    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              eval_metric="rmse", callbacks=[lgb.early_stopping(20)])

    # Прогноз
    y_pred = model.predict(X_test)

    # Метрики
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    smape = 100 * np.mean(
        2 * np.abs(y_pred - y_test) / (np.abs(y_test) + np.abs(y_pred) + 1e-8)
    )

    rmse_scores.append(rmse)
    smape_scores.append(smape)

    print(f"\n=== Fold {fold} ===")
    print(f"RMSE={rmse:.2f}, sMAPE={smape:.2f}%")

print("\n>>> Средние метрики:")
print(f"RMSE={np.mean(rmse_scores):.2f}, sMAPE={np.mean(smape_scores):.2f}%")
