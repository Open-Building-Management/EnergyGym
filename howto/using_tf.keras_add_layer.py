"""using tf.keras.add"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Lambda, Add

# https://keras.io/api/layers/merging_layers/add/

x1 = tf.random.normal((1,2))
x2 = tf.random.normal((1,1))
print(x1)
print(x2)
y = tf.keras.layers.Add()([x1, x2])
print(y)
print(np.argmax(y))


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
