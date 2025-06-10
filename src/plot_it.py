import numpy as np
import matplotlib.pyplot as plot

def plot_it(func, X=None ):
    if X == None or not isinstance(X, np.ndarray): 
        X = np.arange(-5.0, 5.0, 0.1)
    print(X)
    Y = func(X)
    plot.plot(X,Y) 
    plot.show()