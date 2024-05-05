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

def forward_propagation_classification(inputs, weights, biases):
    # Note 1: 
    # The input, weight and bias, needs to be an matrix rather than an array
    # e.g. input = [1, 2, 3], this is not acceptable as its an array, it needs to be a 2d 1 x 3 matrix
    # hence the right way to represent is input = [[1, 2, 3]] as this is an 1 x 3 matrix
    # 
    # reason for doing so, we are required to multiply the input with weight, in order to procced 
    # with forward propagation, but a matrix multiplication can not be done or not feasible when
    # input is array of dimension 1 and weight is an array of dimension 2

    # Note 2:
    # we will be using sigmoid function as activation function except for the output node
    # we will use softmax for the output activation function
    i = 1
    layer_output = [inputs]
    for i, weight in enumerate(weights):
        bias = biases[i + 1]

        # summation of weighted input and addition of bais term
        WIB = np.matmul(inputs, weight) + bias

        # activation function
        if i == len(weights) - 1:
            # activation using softmax
            WIB = np.exp(WIB)
            WIB = WIB/WIB.sum(axis= 1)
            # weighted_input_sum = np.exp(weighted_input_sum - np.max(weighted_input_sum))
            # weighted_input_sum = weighted_input_sum/weighted_input_sum.sum(axis= 1)

        else:
            # activation using sigmoid activation function
            WIB = 1 + np.exp(-1*WIB)
            WIB = 1/WIB
            # weighted_input_sum = -1*weighted_input_sum
            # weighted_input_sum = 1/(1 + np.exp(weighted_input_sum))
        inputs = WIB
        layer_output.append(WIB)
        i += 1
    return layer_output

# w, b = RWP.read_weights_biases()
# i = np.zeros((1, 1))
# i[0][0] = 1
# print(forward_propagation_classification(i, w, b))

def backpropagation_classification(inputs, target, weights, biases, lr_weight = 0.001, lr_bias = 0.001):
    layer_output = forward_propagation_classification(inputs, weights, biases)
    i = len(weights) - 1
    error = None
    final_error = 0
    while i >= 0:
        I = layer_output[i]
        O = layer_output[i + 1]
        W = weights[i]
        B = biases[i + 1]

        dE_by_dW, dE_by_dB = None, None

        if i == len(weights) - 1:
            # the output layer, where the activation function is softmax
            dE_by_dB = target*(O - 1)
            dE_by_dW = np.matmul(I.T, dE_by_dB)

            # for propagating the error backward we need the error (cross entropy error)
            # Note: for calc. of dE_by_dB and dE_by_dW, the error is already incorporated
            #       the above two equations are the final eqn after considering the error
            error = -1*target*np.log(O)
            final_error = np.sum(error)
        else:
            # the output of hidden layer uses sigmoid activation function
            dE_by_dB = error*O*(1 - O)
            dE_by_dW = np.matmul(I.T, dE_by_dB)
        
        # new error to propagte backward
        error = np.matmul(error, weights[i].T)

        # update the weights
        weights[i] -= lr_weight * dE_by_dW

        # bias update
        biases[i + 1] -= lr_bias * dE_by_dB
        i -= 1
    return weights, biases, final_error

# w, b = RWP.read_weights_biases()
# for ele in w:
#     print(ele)
# print("--")
# for ele in b:
#     print(ele)
# i = np.zeros((1, 1))
# t = np.zeros((1, 4))
# t[0][0], t[0][1], t[0][2], t[0][3] = 1, 2, 3, 4
# i[0][0] = 1
# w, b, e = backpropagation_classification(i, t, w, b)
# print("==")
# for ele in w:
#     print(ele)
# print("--")
# for ele in b:
#     print(ele)
# print("--")
# print(e)