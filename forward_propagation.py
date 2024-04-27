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

def forward_propagation(inputs, weights, biases, relu_alpha = 0.01):
    # Note: 
    # The input, weight and bias, needs to be an matrix rather than an array
    # e.g. input = [1, 2, 3], this is not acceptable as its an array, it needs to be a 2d 1 x 3 matrix
    # hence the right way to represent is input = [[1], [2], [3]] as this is an 1 x 3 matrix
    # 
    # reason for doing so, we are required to multiply the input with weight, in order to procced 
    # with forward propagation, but a matrix multiplication can not be done or not feasible when
    # input is array of dimension 1 and weight is an array of dimension 2
    i = 1
    layer_output = [inputs]
    for i, weight in enumerate(weights):
        bias = biases[i + 1]

        # summation of weighted input
        weighted_input_sum = np.matmul(inputs, weight)
        
        # addition of bias
        weighted_input_sum += bias
        
        # activation function
        if i == len(weights) - 1:
            # activation using softmax
            # classification problems
            row, col = weighted_input_sum.shape
            denominator = 0
            for r in range(row):
                for c in range(col):
                    denominator += np.exp(weighted_input_sum[r][c])
            for r in range(row):
                for c in range(col):
                    weighted_input_sum[r][c] = np.exp(weighted_input_sum[r][c])/denominator
        else:
            # activation using leaky ReLU
            row, col = weighted_input_sum.shape
            for r in range(row):
                for c in range(col):
                    weighted_input_sum[r][c] = max(weighted_input_sum[r][c], relu_alpha * weighted_input_sum[r][c])
        inputs = weighted_input_sum
        layer_output.append(weighted_input_sum)
        i += 1
    return layer_output
        

# w, b = RWP.read_weights_biases()
# i = np.zeros((1, 1))
# i[0][0] = 1
# print(forward_propagation(i, w, b))