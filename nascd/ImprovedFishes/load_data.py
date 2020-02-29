import numpy as np
import csv
import random as rd

import os
HERE = os.path.dirname(os.path.abspath(__file__)) # useful to locate data files with respect to this fil
data_dir = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data")
print(f"DATA_PATH: {data_dir}")

np.random.seed(2019)

def load_data() -> tuple:
    """
    Generate data for the fishes problem.

    Return:
        Tuple of Numpy arrays: ``(train_X, train_y), (valid_X, valid_y)``.
    """
    num_fish = {}
    num_fish['Bream'] = 0
    num_fish['Roach'] = 1
    num_fish['Whitefish'] = 2
    num_fish['Parkki'] = 3
    num_fish['Perch'] = 4
    num_fish['Pike'] = 5
    num_fish['Smelt'] = 6


    with open(os.path.join(data_dir, 'Fish.csv'), newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        data = []

        head = True

        for row in spamreader :
            if head :
                head = False
                continue
            data.append([float(num_fish[row[0]]),float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5]),float(row[6])])

        rd.shuffle(data)
        data = np.array(data)

        prop = 0.8
        sample_size = data.shape[0]
        sep_index = int(prop * sample_size)

        train_X = data[:sep_index+1,0:2]
        train_y = data[:sep_index+1,2:]

        valid_X = data[sep_index+1:,0:2]
        valid_y = data[sep_index+1:,2:]

        print(f"train_X shape: {np.shape(train_X)}")
        print(f"train_y shape: {np.shape(train_y)}")
        print(f"valid_X shape: {np.shape(valid_X)}")
        print(f"valid_y shape: {np.shape(valid_y)}")

        return (train_X,train_y), (valid_X, valid_y)








if __name__ == "__main__":
    load_data()


