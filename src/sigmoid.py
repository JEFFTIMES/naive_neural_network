import numpy as np
from src.plot_it import plot_it


def sigmoid(X:np.array) -> np.array:
    """
    return 1.0 / ( 1.0 + exp(-X) )
    """
    return 1.0 / (1.0 + np.exp(-X))

# def sigmoid(X):
#     return np.where(X >= 0, 1 / (1 + np.exp(-X)), np.exp(X) / (1 + np.exp(X)))


if __name__ == '__main__':
    plot_it(sigmoid)