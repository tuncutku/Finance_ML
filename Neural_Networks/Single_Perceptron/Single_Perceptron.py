import numpy as np

from Activation_Function import activation_func

np.random.seed(42)

def generate_Y_data(X_train):

    n_points = X_train.shape[0] 
    Y_train=[]

    for point in range(n_points):
        if X_train[point,1] > -5 * X_train[point,0] + 2:
            Y_train.append(1)
        else:    
            Y_train.append(-1)

    return np.array(Y_train)

def guess(weights, X_train):

    X_train = np.expand_dims(X_train, axis=0)
    X_train_t = np.transpose(X_train)
    init_guess = np.sum(np.matmul(weights, X_train_t))

    return activation_func(init_guess)


class Custom_Perceptron:
    def __init__(self, n_points, n_features, train_test_split, learning_rate):
        
        # Initialization
        self.weights = np.expand_dims(np.random.random((n_features + 1)), axis=0)
        self.n_points = n_points
        self.n_features = n_features
        self.train_test_split = train_test_split
        self.learning_rate = learning_rate
        self.bias = np.expand_dims(np.ones(n_points), axis=1)


    def generate_data(self):
        
        low = -np.ones((self.n_points, self.n_features),'float')
        high = -np.ones((self.n_points, self.n_features),'float')

        X_total = np.random.uniform(-1, 1, (self.n_points, self.n_features))
        X_total = np.concatenate((X_total,self.bias),axis=1)
        Y_total = generate_Y_data(X_total)

        n_train = int(self.n_points * self.train_test_split)
        n_test = self.n_points - n_train
    
        self.X_train = X_total[:n_train,:]
        self.Y_train = Y_total[:n_train]

        self.X_test = X_total[n_train:,:]
        self.Y_test = Y_total[n_train:]


    def train_model(self):

        n_train_points = self.X_train.shape[0]
        Guess_Array_train = []
        Error_Array_train = []

        for point in range(n_train_points):
            X_guess_train = guess(self.weights, self.X_train[point,:])
            error_train = self.Y_train[point] - X_guess_train

            Guess_Array_train.append(X_guess_train)
            Error_Array_train.append(error_train)

            self.weights += error_train * self.X_train[point,:] * self.learning_rate


    def test_model(self):

        n_test_points = self.X_test.shape[0]
        Guess_Array_test = []
        Error_Array_test = []

        for point in range(n_test_points):
            X_guess_test = guess(self.weights, self.X_test[point,:])
            error_test = self.Y_test[point] - X_guess_test

            Guess_Array_test.append(X_guess_test)
            Error_Array_test.append(abs(error_test)/2)

        self.projection = np.array(Guess_Array_test)
        self.test_error = Error_Array_test

