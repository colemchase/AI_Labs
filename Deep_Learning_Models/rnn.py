# RNN Example (Character Prediction)
import tensorflow as tf
import numpy as np

text = "hello world"
chars = sorted(set(text))
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = np.array(chars)

seq_length = 4
x = []
y = []
for i in range(len(text) - seq_length):
    x.append([char2idx[c] for c in text[i:i+seq_length]])
    y.append(char2idx[text[i+seq_length]])

x = tf.keras.utils.to_categorical(x, num_classes=len(chars))
y = tf.keras.utils.to_categorical(y, num_classes=len(chars))

model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, input_shape=(seq_length, len(chars))),
    tf.keras.layers.Dense(len(chars), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x, y, epochs=200, verbose=0)
