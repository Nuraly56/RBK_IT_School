import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())

train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])

test["Fare"] = test["Fare"].fillna(test["Fare"].median())

le = LabelEncoder()  #категориальные признаки
train["Name"] = le.fit_transform(train["Name"])
test["Name"] = le.fit_transform(test["Name"])

train["Sex"] = le.fit_transform(train["Sex"])
test["Sex"] = le.fit_transform(test["Sex"])

train["Ticket"] = le.fit_transform(train["Ticket"])
test["Ticket"] = le.fit_transform(test["Ticket"])

train["Cabin"] = le.fit_transform(train["Cabin"])
test["Cabin"] = le.fit_transform(test["Cabin"])

train["Embarked"] = le.fit_transform(train["Embarked"])
test["Embarked"] = le.fit_transform(test["Embarked"])

train = train.drop(["Name", "Ticket"], axis = 1) #Удаление
test = test.drop(["Name", "Ticket"], axis = 1)

X = train.drop("Survived", axis=1) #Разделение на тренировочные и валидационные
y = train["Survived"]

X_train, X_val, y_train, y_val = train_test_split( #80 на 20
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print("Accuracy is", " ", accuracy)

predictions = model.predict(test)
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})
submission.to_csv("submission.csv", index=False)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [4, 6, 8, None],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(train.head())
print(test.head())

print(test.shape)
print(train.shape)

print(train.dtypes) #Name, Sex, Ticket, Cabin, Embarked - категориальные 
print(test.dtypes)  #Name, Sex, Ticket, Cabin, Embarked - категориальные 

print("Размер X_train:", X_train.shape)
print("Размер X_val:", X_val.shape)
print("Размер y_train:", y_train.shape)
print("Размер y_val:", y_val.shape)

sns.countplot(x="Survived", data=train)
plt.title("Survived")
plt.show()

sns.countplot(x="Sex", hue="Survived", data=train)
plt.title("Survived according to sex")
plt.show()

sns.countplot(x="Pclass", hue="Survived", data=train)
plt.title("Survived according to ticket")
plt.show()

sns.histplot(data=train, x="Age", hue="Survived", bins=30, kde=True, multiple="stack")
plt.title("Survived according to age")
plt.show()

print("Best parameters:", grid.best_params_)
print("Best accuracy:", grid.best_score_)