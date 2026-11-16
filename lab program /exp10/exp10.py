import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
X = np.random.rand(100, 2)
Y = np.random.randint(0, 2, (100, 1))
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        input_shape=(2,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3,
    restore_best_weights=True)
history = model.fit(
    X, Y, epochs=20, batch_size=10,
    validation_split=0.2,
    callbacks=[stop], verbose=0)
print("Training Completed")
print("Final Accuracy:",
      history.history['accuracy'][-1])
plt.plot(history.history['loss'])
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.show()
