from deephyper.benchmark import NaProblem
from nascd.xorandor.load_data import load_data
from nascd.xorandor.search_space import create_search_space

# from deephyper.search.nas.model.preprocessing import stdscaler

Problem = NaProblem(seed=4968214)

Problem.load_data(load_data)

# Problem.preprocessing(stdscaler)

Problem.search_space(create_search_space)

Problem.hyperparameters(
    batch_size=2,
    learning_rate=1.0,
    optimizer="rmsprop",
    num_epochs=2500,
    verbose=0,
    callbacks=dict(
        EarlyStopping=dict(
            monitor="loss", mode="min", verbose=0, patience=5  # or 'val_acc' ?
        )
    ),
)

Problem.loss("binary_crossentropy")  # or 'categorical_crossentropy' ?

Problem.metrics(["binary_accuracy"])  # or 'acc' ?


# def myacc_with_pred(info):
#     from sklearn.metrics import accuracy_score
#     import numpy as np

#     y_pred = np.array(info["y_pred"])
#     y_pred = (y_pred >= 0.5).astype(np.int32)
#     y_true = np.array(info["y_true"], dtype=np.int32)

#     acc = (y_true.flatten() == y_pred.flatten()).astype(int).sum() / len(y_true.flatten())
#     # acc = accuracy_score(y_true, y_pred)

#     return acc


Problem.objective("binary_accuracy__max")  # or 'val_acc__last' ?

Problem.post_training(
    repeat=1,
    num_epochs=3000,
    metrics=["binary_accuracy"],
    callbacks=dict()
    # callbacks=dict(
    #     ModelCheckpoint={
    #         'monitor': 'val_r2',
    #         'mode': 'max',
    #         'save_best_only': True,
    #         'verbose': 1
    #     },
    #     EarlyStopping={
    #         'monitor': 'val_r2',
    #         'mode': 'max',
    #         'verbose': 1,
    #         'patience': 10
    #     },
    #     TensorBoard={
    #         'log_dir':'tb_logs',
    #         'histogram_freq':1,
    #         'batch_size':64,
    #         'write_graph':True,
    #         'write_grads':True,
    #         'write_images':True,
    #         'update_freq':'epoch'
    #     })
)

# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == "__main__":
    print(Problem)
