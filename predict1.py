# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Load dataset
dataset = pd.read_csv("archive/ADANIPORTS.csv")
print(dataset.head())

# Extracting the 'Open' column as the feature for prediction
training_set = dataset['Open'].values.reshape(-1, 1)

# Split the dataset into training and testing sets (80:20 ratio)
train_data, test_data = train_test_split(training_set, test_size=0.2, shuffle=False)

# Save training and testing sets into separate CSV files
pd.DataFrame(train_data, columns=['Open']).to_csv("train_data.csv", index=False)
pd.DataFrame(test_data, columns=['Open']).to_csv("test_data.csv", index=False)

# Feature Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_training_set = scaler.fit_transform(train_data)

# Creating input sequences and labels
X_train, y_train = [], []
for i in range(60, len(scaled_training_set)):
    X_train.append(scaled_training_set[i-60:i, 0])
    y_train.append(scaled_training_set[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM model
regressor = Sequential()
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1))
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
regressor.fit(X_train, y_train, epochs=50, batch_size=32)

# Prepare the test data
scaled_test_data = scaler.transform(test_data)

X_test, y_test = [], []
for i in range(60, len(scaled_test_data)):
    X_test.append(scaled_test_data[i-60:i, 0])
    y_test.append(scaled_test_data[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make predictions
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Visualize the results
plt.plot(test_data, color='red', label='Actual Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()