"""using tf.keras.add"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense  # pylint: disable=E0401

# pylint: disable=C0103

# https://keras.io/api/layers/merging_layers/add/

x_1 = tf.random.normal((1, 2))
x_2 = tf.random.normal((1, 1))
print(x_1)
print(x_2)
x_1_2 = tf.keras.layers.Add()([x_1, x_2])
print(x_1_2)
print(np.argmax(x_1_2))


x = np.array([tf.ones((4,))])
adv_out = Dense(2)
adv = adv_out(x)
print(adv)
v_dense = Dense(50)
v_out = Dense(1)
v = v_dense(x)
v = v_out(v)
print(v)
combine = keras.layers.Add()([v, adv])
print(combine)
print(np.argmax(combine[0]))
