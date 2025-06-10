import numpy as np
from src.plot_it import plot_it


def relu(X:np.ndarray) -> np.ndarray:
    """
    return 0 if x <= 0 else x
    """
    return np.maximum(X, 0)




if __name__ == '__main__':
    plot_it(relu)

