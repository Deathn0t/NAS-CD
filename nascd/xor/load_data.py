import numpy as np

# import os
# HERE = os.path.dirname(os.path.abspath(__file__)) # useful to locate data files with respect to this file

np.random.seed(2018)


def load_data(bin_length: int = 16, sample_size: int = 1000, prop: float = 0.8) -> tuple:
    """
    Generate data for linear function -sum(x_i).

    Args:
        bin_length (int): in [1, 32] size of binary words to apply xor
        sample_size (int): number of samples to generate in total (len(train) + len(valid))
        prop (float): in [0,1] proportion of samples to keep in training set. 1-prop will be the proportion in the validation set.

    Return:
        Tuple of Numpy arrays: ``(train_X, train_y), (valid_X, valid_y)``.
    """

    max_sample_size = 2 ** bin_length
    ratio = sample_size / max_sample_size
    print(f"Max. Sample Size: {max_sample_size}")
    print(f"Ratio sample_size/max_sample_size: {ratio}")

    x = np.random.choice([0, 1], (sample_size, 2, bin_length))
    y = x[:, 0, :] ^ x[:, 1, :]

    sep_index = int(prop * sample_size)
    train_X = x[:sep_index]
    train_y = y[:sep_index]

    valid_X = x[sep_index:]
    valid_y = y[sep_index:]

    train_X0 = np.squeeze(train_X[:, 0, :])
    train_X1 = np.squeeze(train_X[:, 1, :])

    valid_X0 = np.squeeze(valid_X[:, 0, :])
    valid_X1 = np.squeeze(valid_X[:, 1, :])

    print(f"train_X0 shape: {np.shape(train_X0)}, train_X1 shape: {np.shape(train_X1)}")
    print(f"train_y shape: {np.shape(train_y)}")
    print(f"valid_X0 shape: {np.shape(valid_X0)}, valid_X1 shape: {np.shape(valid_X1)}")
    print(f"valid_y shape: {np.shape(valid_y)}")

    # Interface to run training data must me respected
    return ([train_X0, train_X1], train_y), ([valid_X0, valid_X1], valid_y)


if __name__ == "__main__":
    load_data()
