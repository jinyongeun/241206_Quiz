import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

filename = "./data/08_pima-indians-diabetes.data.csv"

column_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = pd.read_csv(filename, names=column_names)

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

scaler = MinMaxScaler()
x_scaler = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaler, y, test_size=0.2, random_state=41)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)

print("-" * 25)
print("Actual Values:", y_test)
print("Predicted Values:", y_pred)
print("-" * 25)
print("Actual:", accuracy)

model = DecisionTreeClassifier(max_depth=1000, min_samples_split=60, min_samples_leaf=5)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)

print("-" * 25)
print("Actual Values:", y_test)
print("Predicted Values:", y_pred)
print("-" * 25)
print("Actual:", accuracy)