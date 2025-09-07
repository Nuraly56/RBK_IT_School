import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor, plot_importance as lgb_plot_importance
from xgboost import XGBRegressor, plot_importance as xgb_plot_importance
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) /
                         (np.abs(y_true) + np.abs(y_pred) + 1e-8))


df = pd.read_csv("flights_demand.csv")
print("Данные:", df.shape)
print(df.head())

df["flight_date"] = pd.to_datetime(df["flight_date"])
df = df.sort_values(["route_id", "flight_date"])


for lag in [7, 14, 30]:
    df[f"lag_{lag}"] = df.groupby("route_id")["tickets_sold"].shift(lag)
for win in [7, 30]:
    df[f"ma_{win}"] = df.groupby("route_id")["tickets_sold"].transform(lambda x: x.rolling(win).mean())

lag_cols = ["lag_7", "lag_14", "lag_30", "ma_7", "ma_30"]
df[lag_cols] = df[lag_cols].fillna(0)

df["dow"] = df["flight_date"].dt.dayofweek
df["month"] = df["flight_date"].dt.month
df["year"] = df["flight_date"].dt.year
df["is_weekend"] = (df["dow"] >= 5).astype(int)
holidays = ["2015-01-01", "2016-01-01", "2017-01-01",
            "2015-03-21", "2016-03-21", "2017-03-21"]
df["is_holiday"] = df["flight_date"].isin(pd.to_datetime(holidays)).astype(int)
df["season"] = (df["month"] % 12) // 3

features = [
    "route_id", "distance_km", "aircraft_capacity",
    "avg_lead_time_days", "dow", "month", "year",
    "lag_7", "lag_14", "lag_30", "ma_7", "ma_30",
    "is_weekend", "is_holiday", "season"
]
X = df[features].astype(float)
y = df["tickets_sold"]

print("\nРазмер X:", X.shape, "Размер y:", y.shape)

models = {
    "LightGBM": LGBMRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=6, num_leaves=31, random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=6, subsample=0.8,
        colsample_bytree=0.8, random_state=42
    )
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
metrics_summary = {}

for name, model in models.items():
    rmses, smapes = [], []
    print(f"\n=== {name} ===")

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        s = smape(y_test.values, y_pred)

        rmses.append(rmse)
        smapes.append(s)

        print(f"Fold {fold+1}: RMSE={rmse:.2f}, sMAPE={s:.2f}%")

    metrics_summary[name] = {"RMSE": rmses, "sMAPE": smapes}
    print(f">>> Средние метрики {name}: RMSE={np.mean(rmses):.2f}, sMAPE={np.mean(smapes):.2f}%")


    if name == "LightGBM":
        lgb_plot_importance(model, max_num_features=15)
        plt.title("LightGBM Feature Importance")
        plt.show()
    elif name == "XGBoost":
        xgb_plot_importance(model, max_num_features=15)
        plt.title("XGBoost Feature Importance")
        plt.show()

