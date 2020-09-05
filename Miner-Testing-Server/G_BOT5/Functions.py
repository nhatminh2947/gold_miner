import numpy as np

import Constants

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def valid (x,y):
    return (x>=0 and x < Constants.MAP_MAX_Y and y>=0 and y < Constants.MAP_MAX_X)