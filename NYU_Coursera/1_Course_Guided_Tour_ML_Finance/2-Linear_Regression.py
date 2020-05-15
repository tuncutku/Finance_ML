import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.python.layers import core as core_layers
try:
    from mpl_toolkits.mplot3d import Axes3D
except: pass

sys.path.append("..")

def reset_graph(seed=42):
    """
    Utility function to reset current tensorflow computation graph
    and set the random seed 
    """
    # to make results reproducible across runs
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def generate_data(n_points=10000, n_features=3, use_nonlinear=True, 
                    noise_std=0.1, train_test_split = 4):
    """
    Arguments:
    n_points - number of data points to generate
    n_features - a positive integer - number of features
    use_nonlinear - if True, generate non-linear data
    train_test_split - an integer - what portion of data to use for testing
    
    Return:
    X_train, Y_train, X_test, Y_test, n_train, n_features
    """
    
    # Linear data or non-linear data?
    if use_nonlinear:
        weights = np.array([[1.0, 0.5, 0.2],[0.5, 0.3, 0.15]])
    else:
        weights = np.array([1.0, 0.5, 0.2])
        
    bias = np.ones(n_points).reshape((-1,1))
    low = - np.ones((n_points,n_features),'float')
    high = np.ones((n_points,n_features),'float')
        
    np.random.seed(42)
    X = np.random.uniform(low=low, high=high)
    
    np.random.seed(42)
    noise = np.random.normal(size=(n_points, 1))
    noise_std = 0.1
    
    if use_nonlinear:
        Y = (weights[0,0] * bias + np.dot(X, weights[0, :]).reshape((-1,1)) + 
             np.dot(X*X, weights[1, :]).reshape([-1,1]) +
             noise_std * noise)
    else:
        Y = (weights[0] * bias + np.dot(X, weights[:]).reshape((-1,1)) + 
             noise_std * noise)
    
    n_test = int(n_points/train_test_split)
    n_train = n_points - n_test
    
    X_train = X[:n_train,:]
    Y_train = Y[:n_train].reshape((-1,1))

    X_test = X[n_train:,:]
    Y_test = Y[n_train:].reshape((-1,1))
    
    return X_train, Y_train, X_test, Y_test, n_train, n_features

X_train, Y_train, X_test, Y_test, n_train, n_features = generate_data(use_nonlinear=False)
print(X_train.shape, Y_train.shape)

def numpy_lin_regress(X_train, Y_train):
    """
    numpy_lin_regress - Implements linear regression model using numpy module
    Arguments:
    X_train  - np.array of size (n by k) where n is number of observations 
                of independent variables and k is number of variables
    Y_train - np.array of size (n by 1) where n is the number of observations of dependend variable
    
    Return:
    np.array of size (k+1 by 1) of regression coefficients
    """
    ### START CODE HERE ### (≈ 3 lines of code)
    # number of features
    observations = X_train.shape[0]
    # add the column of ones
    X_new = np.concatenate((np.ones((observations,1)), X_train), axis = 1)
    theta_numpy = np.linalg.inv(np.dot(X_new.T, X_new)).dot(X_new.T).dot(Y_train).reshape(-1,1)
    # default answer, replace this
    # theta_numpy = np.array([0.] * (ndim + 1)) 
    ### END CODE HERE ###
    return theta_numpy

theta_numpy = numpy_lin_regress(X_train, Y_train)
theta_numpy.squeeze()

def sklearn_lin_regress(X_train, Y_train):
    """
    Arguments:
    X_train  - np.array of size (n by k) where n is number of observations 
                of independent variables and k is number of variables
    Y_train - np.array of size (n by 1) where n is the number of observations of dependend variable
    
    Return:
    np.array of size (k+1 by 1) of regression coefficients
    """ 
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    ### START CODE HERE ### (≈ 3 lines of code)
    # use lin_reg to fit training data
    lin_reg.fit(X_train, Y_train)
    theta_sklearn = np.append(lin_reg.intercept_,lin_reg.coef_).reshape(-1,1)
    ### END CODE HERE ###
    return theta_sklearn

theta_sklearn = sklearn_lin_regress(X_train, Y_train)
theta_sklearn.squeeze()

def tf_lin_regress(X_train, Y_train):
    """
    Arguments:
    X_train  - np.array of size (n by k) where n is number of observations 
                of independent variables and k is number of variables
    Y_train - np.array of size (n by 1) where n is the number of observations of dependend variable
    
    Return:
    np.array of size (k+1 by 1) of regression coefficients
    """
    ### START CODE HERE ### (≈ 7-8 lines of code)
    # add the column of ones
    X_new = np.hstack((np.ones((X_train.shape[0],1)),X_train))
    # define theta for later evaluation
    X = tf.constant(X_new, dtype = tf.float32, name ='X')
    y = tf.constant(Y_train, dtype = tf.float32, name ='y')
    XT = tf.transpose(X)
    theta = tf.matmul(tf.matmul(tf.linalg.inv(tf.matmul(XT,X)),XT),y)
    ### END CODE HERE ###
    with tf.compat.v1.Session() as sess:
        theta_value = theta.eval()
    return theta_value

theta_tf = tf_lin_regress(X_train, Y_train)
theta_tf.squeeze()

class LinRegressNormalEq:
    """
    class LinRegressNormalEq - implements normal equation, maximum likelihood estimator (MLE) solution
    """
    def __init__(self, n_features, learning_rate=0.05, L=0):
        import math as m
        # input placeholders
        self.X = tf.placeholder(tf.float32, [None, n_features], name="X") 
        self.Y = tf.placeholder(tf.float32, [None, 1], name="Y")
    
        # regression parameters for the analytical solution using the Normal equation
        self.theta_in = tf.placeholder(tf.float32, [n_features+1,None])

        # Augmented data matrix is obtained by adding a column of ones to the data matrix
        data_plus_bias = tf.concat([tf.ones([tf.shape(self.X)[0], 1]), self.X], axis=1)
        
        XT = tf.transpose(data_plus_bias)
        
        #############################################
        # The normal equation for Linear Regression
        
        self.theta = tf.matmul(tf.matmul(
            tf.matrix_inverse(tf.matmul(XT, data_plus_bias)), XT), self.Y)
        
        # mean square error in terms of theta = theta_in
        self.lr_mse = tf.reduce_mean(tf.square(
            tf.matmul(data_plus_bias, self.theta_in) - self.Y))
                       
        #############################################
        # Estimate the model using the Maximum Likelihood Estimation (MLE)
        
        # regression parameters for the Maximum Likelihood method
        # Note that there are n_features+2 parameters, as one is added for the intercept, 
        # and another one for the std of noise  
        self.weights = tf.Variable(tf.random_normal([n_features+2, 1]))
        
        # prediction from the model
        self.output = tf.matmul(data_plus_bias, self.weights[:-1, :])

        gauss = tf.distributions.Normal(loc=0.0, scale=1.0)

        # Standard deviation of the Gaussian noise is modelled as a square of the 
        # last model weight
        sigma = 0.0001 + tf.square(self.weights[-1]) 
        
        # though a constant sqrt(2*pi) is not needed to find the best parameters, here we keep it
        # to get the value of the log-LL right 
        pi = tf.constant(m.pi)
    
        log_LL = tf.log(0.00001 + (1/( tf.sqrt(2*pi)*sigma)) * gauss.prob((self.Y - self.output) / sigma ))  
        self.loss = - tf.reduce_mean(log_LL)
        
        self.train_step = (tf.train.AdamOptimizer(learning_rate).minimize(self.loss), -self.loss)

def run_normal_eq(X_train, Y_train, X_test, Y_test, learning_rate=0.05):
    """
    Implements normal equation using tensorflow, trains the model using training data set
    Tests the model quality by computing mean square error (MSE) of the test data set
    
    Arguments:
    X_train  - np.array of size (n by k) where n is number of observations 
                of independent variables and k is number of variables
    Y_train - np.array of size (n by 1) where n is the number of observations of dependend variable
    
    X_test  - np.array of size (n by k) where n is number of observations 
                of independent variables and k is number of variables
    Y_test - np.array of size (n by 1) where n is the number of observations of dependend variable
    
    
    Return a tuple of:
        - np.array of size (k+1 by 1) of regression coefficients
        - mean square error (MSE) of the test data set
        - mean square error (MSE) of the training data set
    """
    # create an instance of the Linear Regression model class  
    n_features = X_train.shape[1]
    model = LinRegressNormalEq(n_features=n_features, learning_rate=learning_rate)

    ### START CODE HERE ### (≈ 10-15 lines of code)
    # train the model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Normal equation for Linear Regression
        theta_value = sess.run(model.theta, feed_dict ={
            model.X: X_train,
            model.Y: Y_train
        })
        lr_mse_train = sess.run(model.lr_mse, feed_dict ={
            model.X: X_train,
            model.Y: Y_train,
            model.theta_in: theta_value
        })
        lr_mse_test = sess.run(model.lr_mse, feed_dict ={
            model.X: X_test,
            model.Y: Y_test,
            model.theta_in: theta_value
        })
             
    ### END CODE HERE ###
    return theta_value, lr_mse_train, lr_mse_test

theta_value, lr_mse_train, lr_mse_test = run_normal_eq(X_train, Y_train, X_test, Y_test)
theta_value.squeeze()

def run_mle(X_train, Y_train, X_test, Y_test, learning_rate=0.05, num_iter=5000):
    """
    Maximum likelihood Estimate (MLE)
    Tests the model quality by computing mean square error (MSE) of the test data set
    
    Arguments:
    X_train  - np.array of size (n by k) where n is number of observations 
                of independent variables and k is number of variables
    Y_train - np.array of size (n by 1) where n is the number of observations of dependend variable
    
    X_test  - np.array of size (n by k) where n is number of observations 
                of independent variables and k is number of variables
    Y_test - np.array of size (n by 1) where n is the number of observations of dependend variable
    
    
    Return a tuple of:
        - np.array of size (k+1 by 1) of regression coefficients
        - mean square error (MSE) of the test data set
        - mean square error (MSE) of the training data set
    """
    # create an instance of the Linear Regression model class  
    n_features = X_train.shape[1]
    model = LinRegressNormalEq(n_features=n_features, learning_rate=learning_rate)
    
    # train the model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Now train the MLE parameters 
        for _ in range(num_iter):
            (_ , loss), weights = sess.run((model.train_step, model.weights), feed_dict={
                model.X: X_train,
                model.Y: Y_train
                })

        # make test_prediction
        Y_test_predicted = sess.run(model.output, feed_dict={model.X: X_test})

        # output std sigma is a square of the last weight
        std_model = weights[-1]**2 
        sess.close()
    return weights[0:-1].squeeze(), loss, std_model

weights, loss, std_model = run_mle(X_train, Y_train, X_test, Y_test)
weights.squeeze()

