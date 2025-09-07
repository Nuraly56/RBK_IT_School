# 1.py — Классификация задержек вылета (is_dep_delay_15)

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

# === 1. Подключение к БД (поменяй порт если нужно) ===
engine = create_engine("postgresql+psycopg2://dwh_user:dwh_pass@localhost:5434/airport_dwh")

# === 2. Загрузка данных ===
query = """
SELECT f.flight_id,
       date(f.scheduled_departure) AS calendar_date,
       EXTRACT(DOW   FROM f.scheduled_departure)::INT AS dow,
       EXTRACT(MONTH FROM f.scheduled_departure)::INT AS month,
       EXTRACT(HOUR  FROM f.scheduled_departure)::INT AS hour,
       r.route_id,
       r.distance_km AS great_circle_km,
       ac.range AS aircraft_capacity,
       tf_sold.tickets_sold_so_far,
       tf_sold.avg_lead_time_days,
       CASE WHEN (f.actual_departure - f.scheduled_departure) > interval '15 minutes' THEN 1 ELSE 0 END AS is_dep_delay_15
FROM ods.flights f
JOIN dim_route r
     ON r.origin_airport_id = (SELECT airport_id FROM dim_airport WHERE airport_code=f.departure_airport)
    AND r.destination_airport_id = (SELECT airport_id FROM dim_airport WHERE airport_code=f.arrival_airport)
JOIN ods.aircrafts ac  ON ac.aircraft_code = f.aircraft_code
LEFT JOIN (
    SELECT f2.flight_id,
           COUNT(*) AS tickets_sold_so_far,
           AVG((f2.scheduled_departure::date - t2.book_date::date)) AS avg_lead_time_days
    FROM ods.flights f2
    JOIN ods.ticket_flights tf2 ON tf2.flight_id=f2.flight_id
    JOIN ods.tickets t2 ON t2.ticket_no=tf2.ticket_no
    GROUP BY f2.flight_id
) tf_sold ON tf_sold.flight_id=f.flight_id;
"""
df = pd.read_sql(query, engine)
print("Данные:", df.shape)
print(df.head())

# === 3. Обработка дат + сортировка ===
df['calendar_date'] = pd.to_datetime(df['calendar_date'])
df = df.sort_values('calendar_date')

# === 4. Историческая пунктуальность (rolling mean по route_id+hour) ===
df['avg_delay_by_route_hour'] = (
    df.groupby(['route_id','hour'])['is_dep_delay_15']
      .transform(lambda x: x.shift().rolling(30, min_periods=1).mean())
)

# === 5. Train/Test split по времени ===
split_date = df['calendar_date'].quantile(0.8)
train = df[df['calendar_date'] <= split_date]
test  = df[df['calendar_date'] >  split_date]

features = ['dow','month','hour','route_id','great_circle_km',
            'aircraft_capacity','tickets_sold_so_far','avg_lead_time_days',
            'avg_delay_by_route_hour']

X_train, y_train = train[features].fillna(0), train['is_dep_delay_15']
X_test,  y_test  = test[features].fillna(0),  test['is_dep_delay_15']

# === 6. Логистическая регрессия (бейзлайн) ===
logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
logreg.fit(X_train, y_train)
pred_proba = logreg.predict_proba(X_test)[:,1]

print("\n=== Logistic Regression ===")
print("ROC-AUC:", roc_auc_score(y_test, pred_proba))
print("PR-AUC:", average_precision_score(y_test, pred_proba))

thresholds = np.linspace(0.1,0.9,9)
f1_scores = [f1_score(y_test, (pred_proba>t).astype(int)) for t in thresholds]
best_t = thresholds[np.argmax(f1_scores)]
print("Best F1:", max(f1_scores), "@ threshold", best_t)

# === 7. LightGBM ===
train_data = lgb.Dataset(X_train, label=y_train)
test_data  = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    "objective": "binary",
    "metric": ["auc","average_precision"],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "is_unbalance": True
}
model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=200, early_stopping_rounds=20)

proba_lgb = model.predict(X_test, num_iteration=model.best_iteration)
print("\n=== LightGBM ===")
print("ROC-AUC:", roc_auc_score(y_test, proba_lgb))
print("PR-AUC:", average_precision_score(y_test, proba_lgb))

thresholds = np.linspace(0.1,0.9,9)
f1_scores = [f1_score(y_test, (proba_lgb>t).astype(int)) for t in thresholds]
best_t = thresholds[np.argmax(f1_scores)]
print("Best F1:", max(f1_scores), "@ threshold", best_t)

# === 8. Интерпретация (Feature Importance + SHAP) ===
lgb.plot_importance(model, max_num_features=10)
plt.show()

explainer = shap.TreeExplainer(model)
sample = X_test.sample(2000, random_state=42)
shap_values = explainer.shap_values(sample)
shap.summary_plot(shap_values, sample)
