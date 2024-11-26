import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
np.random.seed(42)
normal_data = np.random.normal(loc=0, scale=1, size=(1000, 10))
anomalies = np.random.normal(loc=2, scale=2, size=(100, 10))

# Combine and split data
X = np.vstack((normal_data, anomalies))
y = np.hstack((np.zeros(1000), np.ones(100)))  # 0 for normal, 1 for anomaly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the autoencoder model
input_dim = X_train.shape[1]
encoding_dim = 5

input_layer = keras.layers.Input(shape=(input_dim,))
encoder = keras.layers.Dense(encoding_dim, activation="relu")(input_layer)
decoder = keras.layers.Dense(input_dim, activation="sigmoid")(encoder)
autoencoder = keras.models.Model(inputs=input_layer, outputs=decoder)

autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
autoencoder.fit(X_train_scaled, X_train_scaled, 
                epochs=50, 
                batch_size=32, 
                validation_split=0.2,
                verbose=0)

# Predict on test data
predictions = autoencoder.predict(X_test_scaled)
mse = np.mean(np.power(X_test_scaled - predictions, 2), axis=1)

# Set threshold for anomaly detection (e.g., 95th percentile of errors)
threshold = np.percentile(mse, 95)

# Detect anomalies
anomalies_detected = mse > threshold

# Calculate accuracy
accuracy = np.mean(anomalies_detected == y_test)

print(f"Accuracy: {accuracy:.2f}")
print(f"True anomalies detected: {np.sum(anomalies_detected & (y_test == 1))}/{np.sum(y_test == 1)}")
print(f"False positives: {np.sum(anomalies_detected & (y_test == 0))}/{np.sum(y_test == 0)}")

