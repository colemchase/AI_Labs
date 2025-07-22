# LSTM vs GRU Example
import tensorflow as tf
import numpy as np

x = np.random.random((100, 10, 8))
y = np.random.randint(2, size=(100, 1))

# LSTM model
lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(16, input_shape=(10, 8)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy')
lstm_model.fit(x, y, epochs=3)

# GRU model
gru_model = tf.keras.Sequential([
    tf.keras.layers.GRU(16, input_shape=(10, 8)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
gru_model.compile(optimizer='adam', loss='binary_crossentropy')
gru_model.fit(x, y, epochs=3)
