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
        print("input shape: ", np.shape(data[:,0:2]))
        print("output shape: ", np.shape(data[:,2:]))


        return (data[:,0:2], data[:,2:]), (data[:,0:2], data[:,2:])








if __name__ == "__main__":
    load_data()


