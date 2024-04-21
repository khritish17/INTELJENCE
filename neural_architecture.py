'''
    [WARNING] Should be called once, cause it will generate new weight and bias
    if previous weights bias exists, then it will replace them
    
    It builds the neural architecture for the network
'''

import numpy as np

def build_neural_net(neural_architecture):
    
    def validity(): # validity check of the architecture
        layers = len(neural_architecture)
        for i in range(layers):
            ele = neural_architecture[i]
            if type(ele) != type(1):
                print(f"No. of neurons in layer {i + 1} needs to be of an integer value")
                exit()
    
    def generate_weight_matrix(dim_x, dim_y): # generate the weight matrix with the given dimension
        # weights = U(-limit, limit), U: Uniform Distribution
        # limit = sqrt(6/(fan_in + fan_out)) : xavier initialization
        # fan_in: number of incoming edges to the node (number of nodes in the previoius layer)
        # fan_in: number of outgoing edges from the node (number of nodes in the next_layer layer)
        fan_in, fan_out = dim_x, dim_y
        limit = (6/(fan_in + fan_out))**0.5 
        # weight = np.ndarray((dim_x, dim_y))
        weight = np.random.uniform(-limit, limit, (dim_x, dim_y))
        return weight
    
    def generate_bias_vector(layer_node_count): # generate the bias vector for the given layer
        # bias = U(-0.1, 0.1), U: Uniform Distribution
        return np.random.uniform(-0.1, 0.1, layer_node_count)


    # checking the validity of neural architecture
    validity() # if the architecture is not valid, it will close the program

    # create weights and bias
    layers = len(neural_architecture)
    weights = []
    biases = []
    for layer in range(layers - 1):
        # weight matrix dimension for the interface
        dim = [neural_architecture[layer], neural_architecture[layer + 1]]
        weights.append(generate_weight_matrix(dim[0], dim[1]))
    
    for layer in range(layers):
        # bias vector for each node in the layer
        biases.append(generate_bias_vector(neural_architecture[layer]))
