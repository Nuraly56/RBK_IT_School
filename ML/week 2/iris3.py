import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("iris.csv")
print(df.head())

plt.scatter(df["PetalLengthCm"], df["PetalWidthCm"])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, rf_pred))

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_val)
print("kNN Accuracy:", accuracy_score(y_val, knn_pred))

cm = confusion_matrix(y_val, knn_pred)
print("Confusion Matrix:")
print(cm)

sns.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Fact")
plt.show()

new_data = [[5.0, 3.5, 1.3, 0.2],
            [6.5, 3.0, 5.5, 2.0]]
new_df = pd.DataFrame(new_data, columns=X.columns)
new_pred = knn_model.predict(new_df)
print(new_pred)
