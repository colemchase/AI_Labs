# Autoencoder for Dimensionality Reduction
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_digits

X, _ = load_digits(return_X_y=True)
X = MinMaxScaler().fit_transform(X)

inp = Input(shape=(64,))
encoded = Dense(32, activation='relu')(inp)
encoded = Dense(16, activation='relu')(encoded)
decoded = Dense(64, activation='sigmoid')(encoded)

autoencoder = Model(inputs=inp, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X, X, epochs=10, batch_size=32)
