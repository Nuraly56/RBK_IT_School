import pandas as pd
import matplotlib.pyplot as plt

# === 1. Загружаем CSV ===
df = pd.read_csv("C:/Users/Nuraly/Desktop/airport_etl/ml_part/flights_for_ml.csv")

print("Размер данных:", df.shape)
print("\nПервые строки:")
print(df.head())

# === 2. Проверим пропуски ===
print("\nПропуски по колонкам:")
print(df.isna().sum())

# === 3. Распределение целевого признака ===
print("\nРаспределение is_dep_delay_15:")
print(df["is_dep_delay_15"].value_counts(normalize=True))

# Бейзлайн (частота положительного класса)
positive_rate = df["is_dep_delay_15"].mean()
print(f"\nБейзлайн (частота задержек > 15 мин): {positive_rate:.3f}")

# === 4. Визуализация распределения ===
plt.figure(figsize=(5,4))
df["is_dep_delay_15"].value_counts().plot(kind="bar", color=["skyblue","salmon"])
plt.xticks([0,1], ["on-time (0)", "delayed >15min (1)"])
plt.title("Распределение классов (is_dep_delay_15)")
plt.ylabel("Количество рейсов")
plt.show()
