from deephyper.benchmark import NaProblem
from nascd.xorandor.load_data import load_data
from nascd.xorandor.search_space import create_search_space

# from deephyper.search.nas.model.preprocessing import stdscaler

Problem = NaProblem(seed=2019)

Problem.load_data(load_data)

# Problem.preprocessing(stdscaler)

Problem.search_space(create_search_space)

Problem.hyperparameters(
    batch_size=1,
    learning_rate=0.01,
    optimizer="sgd",
    num_epochs=5,
    # callbacks=dict(
    #     EarlyStopping=dict(
    #         monitor='val_r2', # or 'val_acc' ?
    #         mode='max',
    #         verbose=0,
    #         patience=5
    #     )
    # )
)

Problem.loss("binary_crossentropy")  # or 'categorical_crossentropy' ?

Problem.metrics(["acc"])  # or 'acc' ?


def myacc_with_pred(info):
    from sklearn.metrics import accuracy_score
    import numpy as np

    y_pred = np.array(info["y_pred"])
    y_pred = (y_pred >= 0.5).astype(int)
    y_true = info["y_true"]
    return accuracy_score(y_true, y_pred)


Problem.objective(myacc_with_pred)  # or 'val_acc__last' ?

Problem.post_training(
    repeat=1,
    num_epochs=10,
    metrics=["acc"],
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
