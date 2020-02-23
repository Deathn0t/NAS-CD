import numpy as np

# import os
# HERE = os.path.dirname(os.path.abspath(__file__)) # useful to locate data files with respect to this file

np.random.seed(2018)


def load_data() -> tuple:
    """
    Generate data for the xor-and-or problem.

    Return:
        Tuple of Numpy arrays: ``(train_X, train_y), (valid_X, valid_y)``.
    """

    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[a[0] ^ a[1], a[0] and a[1], a[0] or a[1]] for a in x])

    train_X = x[:]
    train_y = y[:]

    valid_X = x[:]
    valid_y = y[:]

    print(f"train_X shape: {np.shape(train_X)}")
    print(f"train_y shape: {np.shape(train_y)}")
    print(f"valid_X shape: {np.shape(valid_X)}")
    print(f"valid_y shape: {np.shape(valid_y)}")

    # Interface to run training data must me respected
    return (train_X, train_y), (valid_X, valid_y)


if __name__ == "__main__":
    load_data()
