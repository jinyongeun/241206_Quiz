import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

filename = "./data/08_pima-indians-diabetes.data.csv"

column_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = pd.read_csv(filename, names=column_names)

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

scaler = MinMaxScaler()
x_scaler = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaler, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)

y_pred_binary = (y_pred > 0.5).astype(int)
print(y_pred_binary)

accuracy = accuracy_score(y_test, y_pred_binary)

print("-" * 25)
print("Actual Values:", y_test)
print("Predicted Values:", y_pred_binary)
print("-" * 25)
print("Actual:", accuracy)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Values', marker='o')
plt.scatter(range(len(y_pred_binary)), y_pred_binary, color='red', label='Predicted Values', marker='x')

plt.title('Comparison of Actual and Predicted Values')
plt.xlabel('Data Index')
plt.ylabel('Class (0 or 1)')
plt.legend()

plt.savefig("./results/linear_regression.png")

# Visualize coefficients
coefficients = model.coef_
feature_names = column_names[:-1]  # Exclude 'class'

plt.figure(figsize=(12, 6))
plt.barh(feature_names, coefficients, color='skyblue')
plt.title('Feature Importance based on Linear Regression Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.axvline(0, color='red', linestyle='--', linewidth=1)
plt.tight_layout()
plt.savefig("./results/feature_coefficients.png")

# Generate prediction equation
intercept = model.intercept_
prediction_equation = "y = {:.3f}".format(intercept)
for i, coef in enumerate(coefficients):
    prediction_equation += " + ({:.3f} * {})".format(coef, feature_names[i])

print("Prediction Equation:")
print(prediction_equation)