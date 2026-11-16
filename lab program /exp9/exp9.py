import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0], [1], [1], [0]])
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(X, Y, epochs=100, verbose=0)
print("Predicted Output:")
print(np.round(model.predict(X)))
plt.plot(history.history['accuracy'])
plt.title("Model Accuracy")
plt.show()
