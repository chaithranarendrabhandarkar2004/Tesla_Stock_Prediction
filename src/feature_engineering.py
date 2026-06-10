import numpy as np


def create_sequences(data, time_step=60):

    X = []
    y = []

    for i in range(len(data) - time_step):

        X.append(data[i:i+time_step])

        y.append(data[i+time_step])

    return np.array(X), np.array(y)