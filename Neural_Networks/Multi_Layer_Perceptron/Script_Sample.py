from NN import Neural_Network
import numpy as np

def genereate_train_data():
    
    X_train = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    Y_train = np.array([[0, 1, 1, 0]])

    return X_train, Y_train

def setup(n_input=2, n_hidden=3, n_output=1, learning_rate=0.9):

    # Generate train data
    X_train, Y_train = genereate_train_data()

    NN = Neural_Network(n_input, n_hidden, n_output, learning_rate)

    np.random.seed(42)

    for _ in range(10000):

        shuffle = np.random.choice([0, 1, 2, 3])
        o_values = NN.feed_forward(np.expand_dims(X_train[shuffle], axis=1))
        NN.train_data(Y_train[0, shuffle], o_values)

    test_values = np.zeros((4,1))

    for idx in range(4):

        test_values[idx] = NN.feed_forward(np.expand_dims(X_train[idx], axis=1))

    return test_values

def visualize():
    pass


if __name__ == "__main__":
    
    test_values = setup()

    visualise()