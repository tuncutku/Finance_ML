import numpy as np
from Activation_Function import sigmoid, d_sigmoid

np.random.seed(42)

def initialize_weights(n_input, n_output):

    column = n_input + 1
    row = n_output
    weights = np.random.random(row * column)
     
    return np.reshape(weights, (row, column))


class Neural_Network():
    
    # TODO Hidden layer input could be dictionary where the user could specify the number of layers 
    # as well as corresponding perceptrons 
    def __init__(self, n_input, n_hidden, n_output, learning_rate):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        # Initiate train data and weights
        self.weights_h = initialize_weights(self.n_input, self.n_hidden)
        self.weights_o = initialize_weights(self.n_hidden, self.n_output)
        self.learning_rate = learning_rate

    def feed_forward(self, X_train):
        '''
        1- Simultaniously use the weights and the inputs to calculate final probabilities.
        2- Train the neural network by adjusting weights.
        '''
        self.X_train = X_train

        v_sigmoid = np.vectorize(sigmoid)

        # Generate weights according to inputs
        # TODO will create a loop to facilitate more layers
        self.Train_1_b = np.concatenate((X_train, np.ones((1,1))), axis=0)
        h_matrix = np.matmul(self.weights_h, self.Train_1_b)
        h_matrix_b = np.concatenate((h_matrix, np.ones((1,1))),axis=0)
        self.h_values_b = v_sigmoid(h_matrix_b)
        o_matrix = np.matmul(self.weights_o, self.h_values_b)
        return v_sigmoid(o_matrix)

    def train_data(self, Y_train, o_values):   

        v_d_sigmoid = np.vectorize(d_sigmoid)

        # Calculate errors
        error_o = Y_train - o_values

        error_h = np.matmul(self.weights_o.T, error_o)

        # Backprobagation
        d_error_2 = v_d_sigmoid(o_values)
        d_weights_o = np.matmul(
            self.learning_rate * np.multiply(error_o, d_error_2), 
            self.h_values_b.T)

        d_error_1 = v_d_sigmoid(self.h_values_b)
        d_weights_h = np.matmul(
            self.learning_rate * np.multiply(error_h, d_error_1), 
            self.X_train.T).T

        # Adjust the weights
        self.weights_h += d_weights_h
        self.weights_o += d_weights_o

