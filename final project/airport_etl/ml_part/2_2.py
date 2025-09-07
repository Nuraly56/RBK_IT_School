import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit

# 1. Загружаем данные
df = pd.read_csv("C:/Users/Nuraly/Desktop/airport_etl/ml_part/flights_demand_ml.csv")

print("Данные:", df.shape)
print(df.head())

# 2. Фичи и таргет
features = [
    "route_id", "distance_km", "aircraft_capacity", "avg_lead_time_days",
    "dow", "month", "year", "lag_7", "lag_14", "lag_30", "ma_7", "ma_30"
]
X = df[features].fillna(0)
y = df["tickets_sold"]

print(f"\nРазмер X: {X.shape} Размер y: {y.shape}")

# 3. TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

rmses, smapes = [], []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # 4. Модель
    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    # 5. Обучение
    model.fit(X_train, y_train)

    # 6. Прогноз
    y_pred = model.predict(X_test)

    # 7. Метрики
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    smape = (100/len(y_test)) * np.sum(
        2 * np.abs(y_pred - y_test) / (np.abs(y_test) + np.abs(y_pred) + 1e-8)
    )

    rmses.append(rmse)
    smapes.append(smape)

    print(f"\n=== Fold {fold+1} ===")
    print(f"RMSE={rmse:.2f}, sMAPE={smape:.2f}%")

print("\n>>> Средние метрики:")
print(f"RMSE={np.mean(rmses):.2f}, sMAPE={np.mean(smapes):.2f}%")
