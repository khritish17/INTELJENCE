import numpy as np
import predefined_algorithm as algo
import neural_architecture as na
import parameters_IO as pio
from matplotlib import pyplot as plt
import os

class Classification_Model:
    def __init__(self, inputs, targets, weights, biases) -> None:
        # Note: self.inputs is the input matrix
        # e.g :
        # [[[1, 2, 3, 4]],  <- is the first input of dimension 1 x 4 (each of these input will be feed into forward propagation)
        #  [[3, 4, 5, 6]],  <- is the second input of dimension 1 x 4 (each of these input will be feed into forward propagation)
        #  [[6, 0, 1, 3]],  <- is the third input of dimension 1 x 4 (each of these input will be feed into forward propagation)
        # ]
        # target:
        # e.g :
        # [[[0, 0, 1]],
        #  [[0, 1, 0]],
        #  [[1, 0, 0]],
        # ]
        self.inputs = inputs
        self.weights = weights
        self.biases = biases
        self.targets = targets
        self.layer_output = [] # to store output of every layer, required in backpropagation
        self.training_inputs, self.training_tragets, self.validation_inputs, self.validation_targets = None, None, None, None
        self.length_training = 0
        self.optimal_weights, self.optimal_biases = self.weights, self.biases
        self.divide_data()

    def divide_data(self, training_percentage = 80):
        '''
            By default it assumes that the whole data will be divided into 80% training and 20% validation
            returns training_inputs, training_tragets, validation_inputs, validation_targets
        '''
        limit = int((training_percentage/100)*len(self.inputs))
        self.length_training = limit
        self.training_inputs = self.inputs[:limit]
        self.training_tragets = self.targets[:limit]
        self.validation_inputs = self.inputs[limit:]
        self.validation_targets = self.targets[limit:]


    def forward_propagation(self, input):
        # input must be one single data point, not the whole input dataset
        # and input must be 1 x n dim matrix and not an array, i.z. [[1, 2, 3]] _/,  [1, 2, 3] X
        self.layer_output = [input] # to store 
        for layer in range(1, len(self.biases)):
            interface = layer - 1
            weight = self.weights[interface]
            bias = self.biases[layer]
            sigma = np.matmul(input, weight) + bias
            if layer == len(self.biases) - 1:
                # last layer: activation is softmax
                output = algo.softmax(sigma)
                self.layer_output.append(output)
                return output
            else:
                # other than last layer: activation is sigmoid
                output = algo.relu(sigma)
                self.layer_output.append(output)
                input = output


    def backpropagation(self, input, target, learning_rates = 0.01):
        output = self.forward_propagation(input)
        n = self.length_training
        # error = (-1*target)*np.log(output)
        # error = (-1 * target*(1/n)) * np.log(output + np.finfo(float).eps)
        error = (1/(2*n))*(output - target)**2
        total_error = np.sum(error)
        
        # move backwards in the layers starting from output to input layer
        dw, db = None, None
        for layer in range(len(self.biases) - 1, 0, -1):
            interface = layer - 1
            
            if layer == len(self.biases) - 1: # output layer
                # softmax activation
                # db = (target/n) * (self.layer_output[layer] - 1)
                # dw = np.matmul(self.layer_output[layer - 1].T, db)
                db = (1/n)*(self.layer_output[layer] - target)*algo.relu_gradient(self.layer_output[layer])
                dw = np.matmul(self.layer_output[layer - 1].T, db)
                
            else: # every other layers except the output layer
                # sigmoid activation
                # db_new = np.matmul(db, self.weights[interface + 1].T)
                # db_new = db_new*(self.layer_output[layer]*(1 - self.layer_output[layer]))
                db_cur = np.matmul(db, self.weights[interface + 1].T) 
                db_cur = db_cur*algo.relu_gradient(self.layer_output[layer])
                db = db_cur
                dw = np.matmul(self.layer_output[layer - 1].T, db)
                pass
            # updating the weights and bias
            self.weights[interface] -= dw * learning_rates
            self.biases[layer] -= db * learning_rates
        return total_error

    def training(self, epochs, learning_rate = 0.01):
        
        tr_error, val_error = [], []
        ep = []
        n = self.length_training
        
        min_validation_error = float('inf')
        optimal_epoch = 1
        for epoch in range(epochs):
            training_error, validation_error = 0, 0
            if epoch % 10 == 0:
                print(f"Epoch #{epoch} !!!")
            for i in range(n):
                # train your model
                input = self.training_inputs[i]
                target = self.training_tragets[i]
                training_error += self.backpropagation(input=input, target=target, learning_rates=learning_rate)
            tr_error.append(training_error/n)

            length_validation_inputs =  len(self.validation_inputs)
            for i in range(length_validation_inputs):
                # validate your model
                input = self.validation_inputs[i]
                target = self.validation_targets[i]
                output = self.forward_propagation(input=input)
                # calculate the validation error
                validation_error += np.sum( (-1*target*(1/n))*np.log(output) )
            val_error.append(validation_error/length_validation_inputs)
            ep.append(epoch + 1)
            
            # adding the mechanism to save the optimal weights and biases that is when
            # the validation error is the minimmum
            if val_error[-1] < min_validation_error:
                min_validation_error = val_error[-1]
                self.optimal_weights = self.weights
                self.optimal_biases = self.biases
                optimal_epoch = epoch
        
        print(f"o_epoch{optimal_epoch}  err:{min_validation_error}")
        
        # print the epoch vs validation error vs training error in the error file inside the metadata directory
        path = os.path.abspath("")+"/metadata"
        try:
            error_data = open(path + "/error_data.txt", "r")
            lines = error_data.readlines()
            last_epoch = int(lines[-1].split(",")[0])
            error_data.close()
            error_data = open(path + "/error_data.txt", "a")
            for i in range(len(ep)):
                error_data.write(f"{ep[i] + last_epoch},{float(tr_error[i])},{float(val_error[i])}\n") # epoch, training error, validation error
            error_data.close()
        except:
            error_data = open(path + "/error_data.txt", "w")
            for i in range(len(ep)):
                error_data.write(f"{ep[i]},{float(tr_error[i])},{float(val_error[i])}\n") # epoch, training error, validation error
            error_data.close()
        
        # plot the training error and validation error wrt epochs
        plt.plot(ep, tr_error, color = 'r')
        plt.plot(ep, val_error, color = 'b')
        plt.plot([optimal_epoch], [min_validation_error], 'o')
        plt.show()
        self.weights = self.optimal_weights
        self.biases = self.optimal_biases
        pio.save_weights_biases(weights=self.weights, biases=self.biases)
        return tr_error, val_error


    def testing(self, testing_inputs, testing_targets):
        n = len(testing_inputs)
        correct_predication = 0
        wrong_prediction = 0
        for i in range(n):
            input = testing_inputs[i]
            target = testing_targets[i]
            output = self.forward_propagation(input= input)
            output = algo.argmax(output)
            if np.sum(target*output) == 1:
                # correct predication
                correct_predication += 1
            else:
                wrong_prediction += 1
            if i % 10 == 0:
                print(f"Testing {round(((i+1)/n)*100, 2)}% complete ")
            #     print(f"correct:{correct_predication}, wrong:{wrong_prediction}, efficiency: {(correct_predication/(i + 1))*100}%")
        print("Total:")
        print(f"correct:{correct_predication}, wrong:{wrong_prediction}, efficiency: {(correct_predication/n)*100}%")


# inputs = []
# targets = []
# arch = [3, 10, 15]
# np.random.seed(10)
# for i in range(100):
#     input = np.random.randint(0, 10, size = (1, arch[0]))
#     target = np.zeros((1, arch[-1]))
#     index = np.random.randint(0, arch[-1])
#     target[0][index] = 1

#     inputs.append(input)
#     targets.append(target)
# # print(targets)
# # na.build_neural_net(arch)
# w, b = pio.read_weights_biases()
# nn = Classification_Model(inputs=inputs, targets=targets, weights=w, biases=b)
# nn.training(500, learning_rate=0.01)
# input = np.random.randint(0, 10, size = (1, arch[0]))
# o = nn.forward_propagation(input= input)
# print(o.shape)
# # print(nn.weights)