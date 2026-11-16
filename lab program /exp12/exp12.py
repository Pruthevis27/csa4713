
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Create sequential data
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

X = []
Y = []

# Prepare dataset
for i in range(len(data)-3):
    X.append(data[i:i+3])
    Y.append(data[i+3])

X = np.array(X).reshape(-1, 3, 1)
Y = np.array(Y)

# Create RNN model
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(10, input_shape=(3, 1)),
    tf.keras.layers.Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X, Y, epochs=100, verbose=0)

# Predict next value
result = model.predict(np.array([[[8], [9], [10]]]))

print("Predicted Next Value:", result[0][0])

# Plot data
plt.plot(data)
plt.title("Sequential Data")
plt.show()
