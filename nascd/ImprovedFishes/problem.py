from deephyper.benchmark import NaProblem
from nascd.ImprovedFishes.load_data import load_data
from nascd.ImprovedFishes.search_space import create_search_space
from deephyper.search.nas.model.preprocessing import minmaxstdscaler

Problem = NaProblem(seed=2019)

Problem.load_data(load_data)

Problem.preprocessing(minmaxstdscaler)

Problem.search_space(create_search_space)

Problem.hyperparameters(
    batch_size=8,
    learning_rate=0.01,
    optimizer='adam',
    num_epochs=200,
    callbacks=dict(
        EarlyStopping=dict(
            monitor='r2', # or 'val_acc' ?
            mode='max',
            verbose=0,
            patience=5
        )
    )
)

Problem.loss('mse') # or 'categorical_crossentropy' ?

Problem.metrics(['r2']) # or 'acc' ?

Problem.objective('r2__max') # or 'val_acc__last' ?

Problem.post_training(
    repeat=1,
    num_epochs=1000,
    metrics=["mse", "r2"],
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
if __name__ == '__main__':
    print(Problem)