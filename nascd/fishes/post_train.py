import numpy as np
from sklearn.metrics import accuracy_score
from deephyper.post.pipeline import train
from nascd.fishes.problem import Problem

config = Problem.space
config["hyperparameters"]["verbose"] = 1

config["arch_seq"] = [0, 13, 7, 0, 0]

config["id"] = 0
config["seed"] = 42
model = train(config)

# data = config["data"]  # loaded in train(config)

# y_pred = np.array(model.predict(data["train_X"]))
# y_pred = (y_pred >= 0.5).astype(int)
# y_true = data["train_Y"]
# for x, pred, obs in zip(data["train_X"], y_pred, y_true):
#     print(x, pred, obs)

# acc = (y_pred.flatten() == y_true.flatten()).astype(int).sum() / len(y_pred.flatten())
# print("acc: ", acc)
# acc = accuracy_score(y_true, y_pred)
# print("acc: ", acc)
