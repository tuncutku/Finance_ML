import numpy as np

from Neural_Networks.Activation_Function import activation_func

np.random.seed(42)

def generate_Y_data(X_data_1, X_data_2):

    if X_data_1 > X_data_2:
        return 1
    else:    
        return -1

class Perceptron:
    def __init__(self, n_points, n_features=2):
        
        # Initialize weights
        self.weights = np.random(1,n_features)


    def generate_data(self, n_points, n_features):
        
        low = -np.ones((n_points,n_features),'float')
        high = -np.ones((n_points,n_features),'float')

        X_train = np.random.uniform(low=low, high=high)

        Y_train = generate_Y_data(X_train)

        return X_train, Y_train


    def train_model(self):

        for idx in range(self.lenght)
            sum_weight += self.weight*