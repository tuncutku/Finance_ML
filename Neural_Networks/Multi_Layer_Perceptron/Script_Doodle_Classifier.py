import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from NN import Neural_Network

def create_data(sample_number = 1000, train_split_ratio= 0.2):

    data_path = "/Users/tuncutku/Desktop/Finance_ML/Neural_Networks/Multi_Layer_Perceptron/Doodle_Raw_Data/"
    raw_rainbow_data = (np.load(data_path + "full-numpy_bitmap-rainbow.npy"))[:sample_number]
    raw_cat_data = (np.load(data_path + "full-numpy_bitmap-cat.npy"))[:sample_number]
    raw_train_data = (np.load(data_path + "full-numpy_bitmap-train.npy"))[:sample_number]

    rainbow_output = np.concatenate((
        np.ones((sample_number, 1)), 
        np.zeros((sample_number, 1)), 
        np.zeros((sample_number, 1))), axis=1
    )

    cat_output = np.concatenate((
        np.zeros((sample_number, 1)),
        np.ones((sample_number, 1)),
        np.zeros((sample_number, 1))), axis=1
    )
    
    train_output = np.concatenate((
        np.zeros((sample_number, 1)), 
        np.zeros((sample_number, 1)),
        np.ones((sample_number, 1))), axis=1
    )

    total_input_data_raw = np.concatenate((raw_rainbow_data, raw_cat_data, raw_train_data), axis=0)
    total_input_data = (total_input_data_raw / 255) / 785
    total_output_data = np.concatenate((rainbow_output, cat_output, train_output), axis=0)

    return train_test_split(
        total_input_data,
        total_output_data,
        test_size=1 - train_split_ratio,
        random_state = 42,
        shuffle=True 
    )


def plot_data(data, n_row = 10, n_column = 10):

    reshaped_data = data.reshape((data.shape[0],28,28))

    sample_image=[]
    fig=plt.figure(figsize=(8, 8))
    for idx in range(n_column * n_row):
        sample_image.append(fig.add_subplot(n_row, n_column, idx+1))
        # sample_image[-1].set_title("Rainbow: " + str(idx))
        plt.imshow(reshaped_data[idx], cmap="Greys")
        sample_image[-1].get_xaxis().set_visible(False)
        sample_image[-1].get_yaxis().set_visible(False)
    plt.show()

if __name__ == "__main__":
    
    sample_number = 1000
    X_train, X_test, Y_train, Y_test = create_data(sample_number)

    learning_rate = 0.8
    n_input = X_train.shape[1]
    n_output = Y_train.shape[1]
    n_hidden = 64
    n_sample = X_train.shape[0]

    NN = Neural_Network(n_input, n_hidden, n_output, learning_rate)

    train_values = np.zeros((2400,3,1))
    
    for idx in range(n_sample):

        o_values = NN.feed_forward(np.expand_dims(X_train[idx], axis=1))
        train_values[idx] = o_values
        NN.train_data(np.expand_dims(Y_train[idx], axis=1), o_values)

    test_values = np.zeros((100,3,1))
    
    for jdx in range(100):

        test_values[jdx] = NN.feed_forward(np.expand_dims(X_train[jdx], axis=1))

    plot_data(X_test, n_row = 10, n_column = 1)

