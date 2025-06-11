import numpy as np

def cross_entropy_error(result: np.ndarray, expected: np.ndarray) -> float:
    """
    Compute the cross entropy error between two matrices, where each column represents a vector.
    Returns a scalar value represents the CEE of all.

    - 1/N * sigma_(i=1)^N[ sigma_(k=1)^K[ expected_j*log(result_j) ] ] 

    Args:
        result (np.ndarray): A matrix where each column is a result vector.
        expected (np.ndarray): A matrix where each column is an expected value vector.

    Returns:
        cee(float): A scalar value.
    """

    if result.shape != expected.shape:
        raise ValueError(f'Mismatched shapes: result {result.shape}, expected {expected.shape}.')
    delta = 1e-7
    return -np.mean(np.sum(expected * np.log(result + delta), axis=0))

def test():

    res, expected = np.array([[0.2,0.2,0.1,0.5],[0.3,0.1,0.1,0.5]]).transpose(), np.array([[0,0,1,0],[0,0,0,1]]).transpose()
    err =  cross_entropy_error(res, expected)
    print(err)


if __name__ == '__main__':
    test()