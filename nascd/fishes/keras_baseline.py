import time
import tensorflow as tf
import numpy as np
from nascd.fishes.load_data import load_data
import matplotlib.pyplot as plt

(x, y), _ = load_data()

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(10, activation=tf.nn.relu)
        self.dense11 = tf.keras.layers.Dense(10, activation=tf.nn.relu)
        self.dense12 = tf.keras.layers.Dense(10, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5)
        # self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense11(x)
        x = self.dense12(x)
        # if training:
        #     x = self.dropout(x, training=training)
        return self.dense2(x)


model = MyModel()

# code to bce loss: https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/keras/backend.py#L4585-L4615
model.compile(loss="mse")

tstart = time.time()
hist  = model.fit(x, y, epochs=1000, batch_size=8).history
t = time.time() - tstart

print(f"Training time: {t}")

plt.plot(hist["loss"])
plt.xlabel("epochs")
plt.ylabel("Loss: MSE")
plt.show()