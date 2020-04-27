import matplotlib.pyplot as plt
import numpy as np

from Perceptron import Custom_Perceptron


def generate_dataset_and_train(n_points, n_features=2, train_test_split=0.8, learning_rate=0.1):
    """
    Train model and generate dataset for 1 layer neural network and split it into training and test portions
    
    """
    perceptron = Custom_Perceptron(n_points, n_features, train_test_split, learning_rate)
    perceptron.generate_data()
    perceptron.train_model()
    perceptron.test_model()

    return perceptron



def plot_results(perceptron):

    def modify_dataset(X, Y):

        index_ones = np.argwhere(Y==1)
        index_zeros = np.argwhere(Y==-1)

        Data_ones = np.concatenate((np.take(X[:,0],index_ones), np.take(X[:,1],index_ones)),axis=1)
        Data_zeros = np.concatenate((np.take(X[:,0],index_zeros), np.take(X[:,1],index_zeros)), axis=1)

        return Data_ones, Data_zeros
    
    train_data_feature_1, train_data_feature_2 = modify_dataset(
        perceptron.X_train, 
        perceptron.Y_train
    )

    test_data_feature_1, test_data_feature_2 = modify_dataset(
        perceptron.X_test, 
        perceptron.projection
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey='row')

    ax1.scatter(train_data_feature_1[:,0],train_data_feature_1[:,1],color='r')
    ax1.scatter(train_data_feature_2[:,0],train_data_feature_2[:,1],color='b')
    ax1.set_xlabel("feature 1")
    ax1.set_ylabel("feature 2")
    ax1.set_title("Train Data")

    ax2.scatter(test_data_feature_1[:,0],test_data_feature_1[:,1],color='r')
    ax2.scatter(test_data_feature_2[:,0],test_data_feature_2[:,1],color='b')
    ax2.set_xlabel("feature 1")
    ax2.set_ylabel("feature 2")
    ax2.set_title("Test Data")

    plt.show()


if __name__ == "__main__":

    n_points = 10000

    perceptron = generate_dataset_and_train(n_points)

    print(perceptron.weights)

    plot_results(perceptron)
