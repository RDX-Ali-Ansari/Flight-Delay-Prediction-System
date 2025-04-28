import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Load dataset
df = pd.read_csv('flight_delays.csv')

# Fill missing values
df['ARRIVAL_DELAY'] = df['ARRIVAL_DELAY'].fillna(0)
df['DEPARTURE_DELAY'] = df['DEPARTURE_DELAY'].fillna(0)
df['CANCELLED'] = df['CANCELLED'].astype(int)

# Streamlit Title
st.title('Flight Delay Analysis')

# Dataset Overview
st.header('Dataset Overview')
st.write('Dataset Shape:', df.shape)
st.write('Dataset Columns:', df.columns)
st.write('Dataset Info:')
st.write(df.info())
st.write('Dataset Statistics:')
st.write(df.describe())

# Histogram Plots
st.header('Delay Distribution')

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.histplot(df['ARRIVAL_DELAY'], kde=True, ax=ax[0])
ax[0].set_title('Distribution of Arrival Delays')
sns.histplot(df['DEPARTURE_DELAY'], kde=True, ax=ax[1])
ax[1].set_title('Distribution of Departure Delays')
st.pyplot(fig)

# Scatter Plot of Arrival vs Departure Delays
st.header('Relationship between Arrival and Departure Delays')
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='DEPARTURE_DELAY', y='ARRIVAL_DELAY', data=df)
ax.set_xlabel('Departure Delay')
ax.set_ylabel('Arrival Delay')
ax.set_title('Relationship between Arrival and Departure Delays')
st.pyplot(fig)

# Probability of flight cancellation
cancelled_flights = df[df['CANCELLED'] == 1]
total_flights = len(df)
cancellation_probability = len(cancelled_flights) / total_flights
st.header('Flight Cancellation Probability')
st.write(f'Probability of flight cancellation: {cancellation_probability:.2f}')

# Linear Regression Model
st.header('Linear Regression Model: Predicting Arrival Delay from Departure Delay')

X = df[['DEPARTURE_DELAY']]
y = df['ARRIVAL_DELAY']
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
st.write(f'Mean Squared Error: {mse:.2f}')
st.write(f'R-squared: {r2:.2f}')

# Plotting the Regression Line
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df['DEPARTURE_DELAY'], df['ARRIVAL_DELAY'])
ax.plot(df['DEPARTURE_DELAY'], y_pred, color='red')
ax.set_xlabel('Departure Delay')
ax.set_ylabel('Arrival Delay')
ax.set_title('Linear Regression Model')
st.pyplot(fig)

# Model Parameters
st.write(f'Intercept: {model.intercept_:.2f}')
st.write(f'Coefficient: {model.coef_[0]:.2f}')
st.write(f'Interpretation: A unit increase in departure delay is associated with a {model.coef_[0]:.2f} unit increase in arrival delay, holding all other factors constant.')
