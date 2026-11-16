
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Create dataset
X = np.array([[[1], [2], [3]],
              [[2], [3], [4]],
              [[3], [4], [5]]])

Y = np.array([4, 5, 6])

# Create models
models = [
    tf.keras.layers.SimpleRNN(5),
    tf.keras.layers.LSTM(5),
    tf.keras.layers.GRU(5)
]

names = ["RNN", "LSTM", "GRU"]
losses = []

# Train and compare
for layer in models:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(3, 1)),
        layer,
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, Y, epochs=20, verbose=0)

    loss = model.evaluate(X, Y, verbose=0)
    losses.append(loss)

    print(names[len(losses)-1], "Loss:", loss)

# Plot comparison
plt.bar(names, losses)
plt.title("RNN, LSTM and GRU Comparison")
plt.ylabel("Loss")
plt.show()
