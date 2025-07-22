# GAN Example (Toy)
import tensorflow as tf
import numpy as np

def make_generator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(2)
    ])
    return model

def make_discriminator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

gen = make_generator()
disc = make_discriminator()
disc.compile(optimizer='adam', loss='binary_crossentropy')

for _ in range(1000):
    real = np.random.normal(0, 1, (16, 2))
    fake_input = np.random.normal(0, 1, (16, 10))
    fake = gen.predict(fake_input, verbose=0)

    x = np.concatenate([real, fake])
    y = np.array([1]*16 + [0]*16)
    disc.train_on_batch(x, y)
