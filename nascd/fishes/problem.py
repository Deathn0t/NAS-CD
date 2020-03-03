from deephyper.benchmark import NaProblem
from nascd.fishes.load_data import load_data
from nascd.fishes.search_space import create_search_space
from deephyper.search.nas.model.preprocessing import minmaxstdscaler

Problem = NaProblem(seed=2019)

Problem.load_data(load_data)

Problem.preprocessing(minmaxstdscaler)

Problem.search_space(create_search_space)

Problem.hyperparameters(
    batch_size=8,
    learning_rate=0.01,
    optimizer="adam",
    num_epochs=200,
    verbose=0,
    callbacks=dict(
        EarlyStopping=dict(
            monitor="r2", mode="max", verbose=0, patience=5  # or 'val_acc' ?
        )
    ),
)

Problem.loss("mse")  # or 'categorical_crossentropy' ?

Problem.metrics(["r2"])  # or 'acc' ?

Problem.objective("r2__max")  # or 'val_acc__last' ?


# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == "__main__":
    print(Problem)
