# Deep Learning Autoencoder
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler

X, _ = load_digits(return_X_y=True)
X = MinMaxScaler().fit_transform(X)

inp = tf.keras.Input(shape=(64,))
encoded = tf.keras.layers.Dense(32, activation='relu')(inp)
encoded = tf.keras.layers.Dense(16, activation='relu')(encoded)
decoded = tf.keras.layers.Dense(64, activation='sigmoid')(encoded)

autoencoder = tf.keras.Model(inp, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X, X, epochs=10, batch_size=32)
