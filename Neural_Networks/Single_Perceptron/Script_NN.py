import matplotlib.pyplot as plt
import numpy as np

from Perceptron import Custom_Perceptron


def generate_dataset_and_train(n_points, n_features=2, train_test_split=0.8, learning_rate=0.6):
    """
    Train model and generate dataset for 1 layer neural network and split it into training and test portions
    
    """
    perceptron = Custom_Perceptron(n_points, n_features, train_test_split, learning_rate)
    perceptron.generate_data()
    perceptron.train_model()
    perceptron.test_model()

    return perceptron



def plot_results(perceptron):

    def modify_dataset(X, Y, *args):

        index_ones = np.argwhere(Y==1)
        index_zeros = np.argwhere(Y==-1)

        Data_ones = np.concatenate((np.take(X[:,0],index_ones), np.take(X[:,1],index_ones)),axis=1)
        Data_zeros = np.concatenate((np.take(X[:,0],index_zeros), np.take(X[:,1],index_zeros)), axis=1)
        
        if args:  
            error_loc = np.argwhere(np.array(Y) != args)[:,1]
            error_loc = np.expand_dims(error_loc, axis=1)
            Data_errors = np.concatenate((np.take(X[:,0],error_loc), np.take(X[:,1],error_loc)),axis=1)

            return Data_ones, Data_zeros, Data_errors

        else:

            return Data_ones, Data_zeros

    
    train_data_feature_1, train_data_feature_2 = modify_dataset(
        perceptron.X_train, 
        perceptron.Y_train
    )

    test_data_feature_1, test_data_feature_2, test_data_errors = modify_dataset(
        perceptron.X_test, 
        perceptron.projection,
        perceptron.Y_test
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey='row')

    ax1.scatter(train_data_feature_1[:,0],train_data_feature_1[:,1],color='g')
    ax1.scatter(train_data_feature_2[:,0],train_data_feature_2[:,1],color='b')
    ax1.set_xlabel("feature 1")
    ax1.set_ylabel("feature 2")
    ax1.set_title("Train Data")

    ax2.scatter(test_data_feature_1[:,0],test_data_feature_1[:,1],color='g')
    ax2.scatter(test_data_feature_2[:,0],test_data_feature_2[:,1],color='b')
    ax2.set_xlabel("feature 1")
    ax2.set_ylabel("feature 2")
    ax2.set_title("Test Data")

    ax2.scatter(test_data_errors[:,0],test_data_errors[:,1],color='r')

    plt.show()


if __name__ == "__main__":

    n_points = 1000

    perceptron = generate_dataset_and_train(n_points)

    aim = perceptron.weights[0][1]/perceptron.weights[0][0]
    bias = perceptron.weights[0][2]
    n_error = sum(perceptron.test_error)
    error_rate = n_error/perceptron.Y_test.size

    print("Equation is: Y = {}*X + {}".format(round(aim,3), round(bias,3)))
    print("Total number of error is: {}".format(n_error))
    print("Error rate is: {}".format(error_rate))

    plot_results(perceptron)
