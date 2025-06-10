import numpy as np

def numerical_gradient(func, vars:np.ndarray) -> np.ndarray: 
    '''
    calculate function func's gradient of the variable matrix vars.

    receives:
    func(function): loss function
    vars(np.ndarray): variables in a matrix (weights) or a column vector (bias)
    
    returns:
    grads(np.ndarray): A matrix contains the func's gradient of the variable matrix or the column vector.
    
    '''

    # a small change
    h = 1e-4

    # a matrix or a column vector
    if len(vars.shape) != 2:
        raise ValueError(f'Matrix shape: {vars.shape}, it is not a 2D-array')



    h = 1e-4
    grads = np.zeros_like(vars)
    for idx in np.ndindex(vars.shape):
        orig = vars[idx]
        vars[idx] = orig + h
        u_bound = func(vars)
        vars[idx] = orig - h
        l_bound = func(vars)
        grads[idx] = (u_bound - l_bound) / (2 * h)
        vars[idx] = orig
    return grads
