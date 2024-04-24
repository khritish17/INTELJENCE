import forward_propagation as fp
import numpy as np

def backpropagation(input, target):
    # 1. get the forward propagation output
    layer_output = fp.forward_propagation(input)
    i = len(layer_output) - 1
    output = layer_output[i]

    # converting list to numpy array for leaverising numpy maths power
    output = np.array(output)
    target = np.array(target)
    
    # 2. Compute the error, e and error function E
    error = target - output
    se_error = error ** 2 # squared error = (target- error)^2
        
    # 3. Update weight through gradient descent
    # 4. Propagate the error to the next layer
    # (note here next layer means the previous 
    # layer as we are traversing in the reverse direction of the network)
    pass
backpropagation([1], [1, 2, 3, 4])