import forward_propagation as fp
import numpy as np
import read_write_parameters as RWP

def backpropagation(input, target, weights, biases):
    
    layer_output, relu_alpha = fp.forward_propagation(input, weights, biases)
    i = len(layer_output) - 1
    e = -1*(np.array(target) - np.array(layer_output[i]))

    def get_relu_derivative(outputs, relu_alpha):
        derivatives = []
        for ele in outputs:
            if ele >= 0:
                derivatives.append(1)
            else:
                derivatives.append(relu_alpha)
        return derivatives

    def transpose_1d(arr):
        l = len(arr)
        # we need to convert it to a l x 1 array
        transposed_array = np.zeros((l, 1))
        for i in range(l):
            transposed_array[i][0] = arr[i]
        return transposed_array

    def convert_1d_to_2d(arr):
        l = len(arr)
        converted_array = np.zeros((1, l))
        for i in range(l):
            converted_array[0][i] = arr[i]
        return converted_array


    while i >= 1:
        I = np.array(layer_output[i - 1]) # input for the current interface
        O = np.array(layer_output[i]) # output for the current interface
        weight = weights[i]
        # dE/dw = I_transpose x (-1.e.F) = I_transpose x M, where M = (-1.e.F)
        F = get_relu_derivative(O, relu_alpha)
        M = e*F
        input_transposed = transpose_1d(I) 
        converted_M = convert_1d_to_2d(M)

        dE_by_dW = np.matmul(input_transposed, converted_M)
        dE_by_db = converted_M
        print(dE_by_db)
        print(dE_by_dW)
        e = np.matmul(e, weight.T)
        i -= 1

w, b = RWP.read_weights_biases()
backpropagation([1], [1, 2, 3, 4], w, b)