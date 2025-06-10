import numpy as np

def softmax(arr:np.ndarray) -> np.ndarray:
    '''
    arr_ji: j-th component of i-th pre-activation 
    a = exp(arr_ji) / sigma_k(exp(arr_ki))
    '''
    axis = 0 # only accept column vector and matrix, calculating by column

    C = np.max(arr, axis=axis, keepdims=True)  # Max per sample
    exp_arr = np.exp(arr - C)
    sum_exp_arr = np.sum(exp_arr, axis=axis, keepdims=True)  # Sum per column
    return exp_arr / sum_exp_arr


def test_col_vector():
    col_v = np.array([1,2,5,4,3,2,1]).reshape(-1, 1)
    soft_col_v = softmax(col_v)
    print(soft_col_v)

def test_row_vector():
    col_v = np.array([1,2,5,4,3,2,1]).reshape(1, -1)
    soft_col_v = softmax(col_v)
    print(soft_col_v)

def test_matrix():
    m = np.random.randn(4,6)
    soft_m = softmax(m)
    print(soft_m)

if __name__ == '__main__':
    test_col_vector()
    test_row_vector()
    test_matrix()