import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('flight_delays.csv')

df['ARRIVAL_DELAY'] = df['ARRIVAL_DELAY'].fillna(0)
df['DEPARTURE_DELAY'] = df['DEPARTURE_DELAY'].fillna(0)
df['CANCELLED'] = df['CANCELLED'].astype(int)

print('Dataset Shape:', df.shape)
print('Dataset Columns:', df.columns)
print('Dataset Info:', df.info())
print('Dataset Statistics:', df.describe())

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df['ARRIVAL_DELAY'], kde=True)
plt.title('Distribution of Arrival Delays')
plt.subplot(1, 2, 2)
sns.histplot(df['DEPARTURE_DELAY'], kde=True)
plt.title('Distribution of Departure Delays')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x='DEPARTURE_DELAY', y='ARRIVAL_DELAY', data=df)
plt.xlabel('Departure Delay')
plt.ylabel('Arrival Delay')
plt.title('Relationship between Arrival and Departure Delays')
plt.show()

cancelled_flights = df[df['CANCELLED'] == 1]
total_flights = len(df)
cancellation_probability = len(cancelled_flights) / total_flights
print(f'Probability of flight cancellation: {cancellation_probability:.2f}')

X = df[['DEPARTURE_DELAY']]
y = df['ARRIVAL_DELAY']
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

plt.figure(figsize=(8, 6))
plt.scatter(df['DEPARTURE_DELAY'], df['ARRIVAL_DELAY'])
plt.plot(df['DEPARTURE_DELAY'], y_pred, color='red')
plt.xlabel('Departure Delay')
plt.ylabel('Arrival Delay')
plt.title('Linear Regression Model')
plt.show()

print(f'Intercept: {model.intercept_:.2f}')
print(f'Coefficient: {model.coef_[0]:.2f}')
print('Interpretation: A unit increase in departure delay is associated with a {model.coef_[0]:.2f} unit increase in arrival delay, holding all other factors constant.')
