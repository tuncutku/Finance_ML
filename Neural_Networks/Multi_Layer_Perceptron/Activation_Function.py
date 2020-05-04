import math
import numpy as np

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def d_sigmoid(y):
    return np.array(y * (1-y))
