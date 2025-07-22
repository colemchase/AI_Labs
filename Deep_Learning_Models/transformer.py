# Mini Transformer (Encoder-only)
import tensorflow as tf

inputs = tf.keras.Input(shape=(None,), dtype="int32")
embedding_layer = tf.keras.layers.Embedding(10000, 64)(inputs)
attention_output = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=64)(embedding_layer, embedding_layer)
outputs = tf.keras.layers.GlobalAveragePooling1D()(attention_output)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(outputs)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam", loss="binary_crossentropy")
model.summary()
