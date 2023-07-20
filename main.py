import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import yfinance as yf
import requests
import matplotlib.pyplot as plt


# Load data into pandas DataFrame
df = pd.read_csv('AUDUSD-rates.csv')

df = df.dropna()

pd.set_option('display.precision', 5)

# Define x and y axis
y = df['Adj Close']
X = df.drop(columns=['Date', 'Adj Close'])

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Applying the linear regression model to make a prediction
lr_model_train_pred = lr_model.predict(X_train)
lr_model_test_pred = lr_model.predict(X_test)

# Evaluate model performance
lr_model_train_mse = mean_squared_error(y_train, lr_model_train_pred)
lr_model_train_r2 = r2_score(y_train, lr_model_train_pred)

lr_model_test_mse = mean_squared_error(y_test, lr_model_test_pred)
lr_model_test_r2 = r2_score(y_test, lr_model_test_pred)

print('LR Model MSE (Train):', lr_model_train_mse)
print('LR Model R2 (Train)', lr_model_train_r2)
print('LR Model MSE (Test):', lr_model_test_mse)
print('LR Model R2 (Test)', lr_model_test_r2)

lr_model_results = pd.DataFrame(['Linear Regression', lr_model_train_mse, lr_model_train_r2, lr_model_test_mse, lr_model_test_r2]).transpose()
lr_model_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']

print(lr_model_results)

# print(lr_model_train_pred)
# print(lr_model_test_pred)

# Training random forrest model
rf_model = RandomForestRegressor(max_depth=2, random_state=42)
rf_model.fit(X_train, y_train)

# Applying the random forrest model to make a prediction
rf_model_train_pred = rf_model.predict(X_train)
rf_model_test_pred = rf_model.predict(X_test)

# Evaluate model performance
rf_model_train_mse = mean_squared_error(y_train, lr_model_train_pred)
rf_model_train_r2 = r2_score(y_train, lr_model_train_pred)

rf_model_test_mse = mean_squared_error(y_test, lr_model_test_pred)
rf_model_test_r2 = r2_score(y_test, lr_model_test_pred)

print('RF Model MSE (Train):', rf_model_train_mse)
print('RF Model R2 (Train)', rf_model_train_r2)
print('RF Model MSE (Test):', rf_model_test_mse)
print('RF Model R2 (Test)', rf_model_test_r2)

rf_model_results = pd.DataFrame(['Linear Regression', rf_model_train_mse, rf_model_train_r2, rf_model_test_mse, rf_model_test_r2]).transpose()
rf_model_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']

print(rf_model_results)

# plt.scatter(x=y_train, y=lr_model_train_pred, alpha=0.3)
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.title('Actual vs. Predicted')
# plt.show()

print(f'The number of rows in the X training data is: {X_train.shape[0]}')
print(f'The number of rows in the y training data is: {y_train.shape[0]}')

# Use trained model to make a prediction
next_instance = pd.DataFrame({'rba-cash-rate':  [4.10], 'fed-reserve-rate': [5.08]})
predicted_rate = lr_model.predict(next_instance)

print(f'The predicted rate for the next day is {predicted_rate}')
