import numpy as np

def mean_squared_error(result: np.ndarray, expected: np.ndarray) -> float:
    """
    Compute the mean squared error between two matrices, where each column represents a vector.
    Returns a scalar value represents the MSE of all.

    Args:
        result (np.ndarray): A matrix where each column is a result vector.
        expected (np.ndarray): A matrix where each column is an expected value vector.

    Returns:
        mse(float): A scalar value represents the MSE.
    """
    if result.shape != expected.shape:
        raise ValueError(f'Mismatched shapes: result {result.shape}, expected {expected.shape}.')

    # Compute the element-wise squared difference
    squared_diff = (result - expected) ** 2

    # Sum up the element-wise squared difference for each column, then calculate MEAN for all columns 
    mse = np.mean(np.sum(squared_diff, axis=0))

    return mse
    
    
def test():

    res, expected = np.array([[0.2,0.2,0.1,0.5],[0.3,0.1,0.1,0.5]]).transpose(), np.array([[0,0,1,0],[0,0,0,1]]).transpose()
    err =  mean_squared_error(res, expected)
    print(err)



if __name__ == '__main__':
    test()