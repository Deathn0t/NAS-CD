import tensorflow as tf
import numpy as np
from nascd.xorandor.load_data import load_data

(x, y), _ = load_data()


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(10, activation=tf.nn.relu)
        self.dense11 = tf.keras.layers.Dense(10, activation=tf.nn.relu)
        self.dense12 = tf.keras.layers.Dense(10, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(3, activation=tf.nn.sigmoid)
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
model.compile(loss="binary_crossentropy", metrics=[tf.keras.metrics.binary_accuracy])

model.fit(x, y, epochs=200, batch_size=2)

y_pred = np.array(model.predict(x))
sig = lambda x: 1 / (1 + np.exp(-x))
y_pred = (y_pred >= 0.5).astype(int)
y_true = np.array(y, dtype=np.int32)

acc = (y_true.flatten() == y_pred.flatten()).astype(int).sum() / len(y_true.flatten())
print(f"acc: {acc}")

for yp, yt in zip(y_pred, y_true):
    print(yp, yt)
