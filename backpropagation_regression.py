import fp_regression as fp
import read_write_parameters as RWP
import numpy as np
import copy as cp

def backpropagation_regression(inputs, target, weights, biases, relu_alpha = 0.01, lr_weight = 0.01, lr_bias = 0.01):
    layer_output = fp.forward_propagation_regression(inputs, weights, biases)
    error = target - layer_output[-1]

    i = len(weights) - 1
    while i >= 0:
        I = layer_output[i]
        O = layer_output[i + 1]
        W = weights[i]
        B = biases[i + 1]

        dE_by_db = None
        dE_by_dW = None
        
        # for ReLU activation function
        # dE/db = (-1.e.F)
        F = cp.deepcopy(O)
        row, col = F.shape
        for r in range(row):
            for c in range(col):
                F[r][c] = 1 if F[r][c] >= 0 else relu_alpha
        dE_by_db = -1*error*F

        # dE/dw = I_transposed x (-1.e.F) = I_transposed x dE/db
        I_transposed = I.T
        dE_by_dW = np.matmul(I_transposed, dE_by_db)
        
        # error propagation to the next layer (in the reverse direction of the network)
        error = np.matmul(error, weights[i].T)

        # weight update
        weights[i] -= lr_weight * dE_by_dW 

        # bias update
        biases[i + 1] -= lr_bias * dE_by_db
        i -= 1
    return weights, biases


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
# w, b = backpropagation_regression(i, t, w, b)
# print("=====")
# for ele in w:
#     print(ele)
# print("--")
# for ele in b:
#     print(ele)
