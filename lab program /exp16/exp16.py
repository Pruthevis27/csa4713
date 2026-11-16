import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
X = np.array([[[1], [2], [3]],
              [[2], [3], [4]],
              [[3], [4], [5]]])
Y = np.array([4, 5, 6])
rnn = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(5, input_shape=(3, 1)),
    tf.keras.layers.Dense(1)
])
lstm = tf.keras.Sequential([
    tf.keras.layers.LSTM(5, input_shape=(3, 1)),
    tf.keras.layers.Dense(1)
])
for model in [rnn, lstm]:
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, Y, epochs=20, verbose=0)
print("RNN Prediction:", rnn.predict(X, verbose=0))
print("LSTM Prediction:", lstm.predict(X, verbose=0))
plt.bar(["RNN", "LSTM"],
        [rnn.evaluate(X, Y, verbose=0),
         lstm.evaluate(X, Y, verbose=0)])
plt.title("RNN vs LSTM")
plt.ylabel("Loss")
plt.show()
