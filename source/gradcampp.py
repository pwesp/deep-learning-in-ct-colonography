import tensorflow as tf

def model_modifier(m):
    m.layers[-1].activation = tf.keras.activations.linear
    return m