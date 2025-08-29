import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("iris.csv")
print(df.head())

sns.scatterplot(data=df, x="PetalLengthCm", y="PetalWidthCm", hue="Species")
plt.show()

X = df.drop(["Id", "Species"], axis=1)  
y = df["Species"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_val)
knn_accuracy = accuracy_score(y_val, knn_pred)
knn_conf_matrix = confusion_matrix(y_val, knn_pred)

print("kNN Accuracy:", knn_accuracy)
print("kNN Confusion Matrix:")
print(knn_conf_matrix)

plt.figure(figsize=(6, 4))
sns.heatmap(knn_conf_matrix, annot=True, fmt='d', cmap='Reds',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('kNN Confusion Matrix')
plt.show()

new_data = [
    [5.0, 3.5, 1.3, 0.2],  
    [6.5, 3.0, 5.5, 2.0]   
]
new_df = pd.DataFrame(np.array(new_data), columns=X.columns)
new_predictions = model.predict(new_df)

print(new_predictions)