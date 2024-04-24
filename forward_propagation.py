import numpy as np
import os 
import read_write_parameters as RWP

# read the neural architecture from metadata.txt
path = os.path.abspath("") + "/metadata"
neural_architecture = []
with open(path + "/metadata.txt", "r") as mf:
    temp = mf.readline()
    temp = temp.split(":")
    neural_architecture = temp[-1].split(",")
neural_architecture = [int(ele) for ele in neural_architecture]


def forward_propagation(inputs, w, b, relu_alpha = 0.01): # input, weights, biases
    
    # weighted inputs
    def weighted_input(inp, weight): # inp is the input to the layer and weight is the weights of that interface
        return np.matmul(inp, weight)
    
    def leakyReLU(arr):
        ans = []
        # relu_alpha = 0.01
        for ele in arr:
            ans.append(max(relu_alpha*ele, ele))
        return ans
    
    # read the weights and biases from metadata folder
    # w, b = RWP.read_weights_biases()
    i = 1

    layer_output = []
    for weight in w:
        # perform the weighted input summation
        weight_input = weighted_input(inputs, weight)
        
        # add the bias term
        bias = b[i]
        i += 1
        weight_input += bias
        
        # perform activation
        inputs = leakyReLU(weight_input)
        layer_output.append(inputs)
    return layer_output
# w, b = RWP.read_weights_biases()
# print(forward_propagation([1], w, b))