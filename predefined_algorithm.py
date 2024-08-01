import numpy as np
from matplotlib import pyplot as plt

def sigmoid(array):
    sigmoid_array = 1/(1 + np.exp(-1*array))
    return sigmoid_array

def relu(array):
    relu = np.maximum(array, 0)
    return relu

def relu_gradient(array):
    relu_grad = np.where(array > 0, 1, 0)
    return relu_grad

def sigmoid_gradient(array):
    sig = sigmoid(array)
    return sig*(1 - sig)

def softmax(array):
    exp_array = np.exp(array)
    # softmax_array = exp_array/np.sum(exp_array) 
    softmax_array = exp_array / (np.sum(exp_array) + np.finfo(float).eps)
    return softmax_array

def gradient_softmax(array):
    soft = softmax(array)
    return soft*(1 - soft)

def argmax(array):
    max_ele = np.max(array)
    for i in range(array.shape[1]):
        if array[0][i] != max_ele:
            array[0][i] = 0
        else:
            array[0][i] = 1
    return array



# a = np.zeros((1, 4))
# a[0][0] = 0.235
# a[0][1] = -1.25
# a[0][2] = 0
# a[0][3] = 0.124
# r = relu(a)
# print(r)
# print(relu_gradient(r))
# print(argmax(a))



# plot for sigmoid and gradient of sigmoid
# x = np.array([i*0.1 for i in range(-100, 101)])
# y = sigmoid(x)
# dy = sigmoid_gradient(x)

# plot for softmx and gradient of softmax
# x = np.array([i*0.1 for i in range(-20, 21)])
# y = softmax(x)
# dy = gradient_softmax(x)

# plt.plot(x, y, x, dy)
# plt.show()