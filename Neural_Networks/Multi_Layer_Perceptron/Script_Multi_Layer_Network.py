from NN import Neural_Network

def setup(n_input=2, n_hidden=2, n_output=2):

    dummy = Neural_Network(n_input, n_hidden, n_output)
    a=dummy.feed_forward()

    return dummy

def visualize():
    pass


if __name__ == "__main__":
    
    dummy = setup()

    visualise()