import numpy as np
from Activation_Function import sigmoid

np.random.seed(42)

def genereate_train_data():
    
    X_train = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    Y_train = np.array([0, 1, 1, 0])

    return X_train, Y_train

def initialize_weights(n_input, n_output):

    column = n_input + 1
    row = n_output
    weights = np.random.random(row * column)
     
    return np.reshape(weights, (row, column))


class Neural_Network():
    
    # TODO Hidden layer input could be dictionary where the user could specify the number of layers 
    # as well as corresponding perceptrons 
    def __init__(self, n_input, n_hidden, n_output):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        # Initiate train data and weights
        self.X_train, self.Y_train = genereate_train_data()
        self.weights_1 = initialize_weights(self.n_input, self.n_hidden)
        self.weights_2 = initialize_weights(self.n_hidden, self.n_output)

    def feed_forward(self):
        '''
        Simultaniously use the weights and the inputs to calculate final probabilities.
        '''
        
        # Generate weights according to inputs
        # TODO will create a loop to facilitate more layers
        Train_1 = np.concatenate((self.X_train[0,:],np.ones(1)),axis=0)
        h_matrix = np.matmul(self.weights_1, Train_1)
        h_values = list(map(sigmoid, h_matrix))

        Train_2 = np.concatenate((h_values,np.ones(1)),axis=0)
        output_matrix = np.matmul(self.weights_2, Train_2)
        o_values = list(map(sigmoid, output_matrix))

        return o_values

    def train():
        pass