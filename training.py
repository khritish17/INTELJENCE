import classification_model as cm
import numpy as np
from matplotlib import pyplot as plt
import read_write_parameters as rwp

class NeuralNetworkTrain:
    def __init__(self) -> None:
        self.classification = False
        self.regression = False
    
    def choose_model(self, classification = False, regression = False):
        if classification ^ regression == False:
            print("Choose one of the models")
        else:
            self.classification, self.regression = classification, regression
    
    def train_model(self, inputs, target, weights, biases, epochs = 100, lr_weight = 0.01, lr_bias = 0.01):
        if self.classification ^ self.regression == False:
            print("Choose a model first!!!")
        else:
            # here the inputs are target are expected, in 2d array/list format
            # devide the inputs data into testing and validation data
            # 80% training and 20% validation data
            # Note: Make sure that the for an input there is a corrosponding target:
            # i.z. rows in inputs should be equal to rows in target, cols of the input matrix need not match the cols of traget matrix
            if inputs.shape[0] != target.shape[0]:
                print("For an input there needs to be a corrosponding target, i.z. rows in inputs should be equal to rows in target, cols of the input matrix need not match the cols of traget matrix")
                exit()
            row, col = inputs.shape
            limit = int(0.8*row)

            # training data
            training_input = inputs[:limit]
            training_target = target[:limit]
            training_error = 0
            # validation data
            validation_input = inputs[limit:]
            validation_target = target[limit:]
            validation_error = 0

            trained_weights, trained_biases = None, None

            if self.classification:
                train_e = [] # list to save the training error at each epochs
                validation_e = [] # list to save the validation error at each epochs
                epoch = [] # to save the epoch
                for e in range(epochs):
                    # training the neural network 
                    for i in range(training_input.shape[0]):
                        input_data = training_input[i:i+1]
                        target_data = training_target[i:i+1]
                        updated_weights, updated_biases, error = cm.backpropagation_classification(input_data, target_data, weights, biases, lr_weight = 0.001, lr_bias = 0.001)
                        trained_weights = updated_weights
                        trained_biases = updated_biases
                        weights, biases = updated_weights, updated_biases
                        # print(weights)
                        training_error += error # remember for classification model we use croos entropy error function
                    
                    # validation of the neural network
                    for i in range(validation_input.shape[0]):
                        input_data = validation_input[i:i+1]
                        target_data = validation_target[i:i+1]
                        layer_output = cm.forward_propagation_classification(inputs= input_data, weights= trained_weights, biases= trained_biases)
                        O = layer_output[-1]
                        # if everything done correctly then the dimension of O should match with target
                        clipped_output = np.clip(O, 1e-300, 1 - 1e-300)
                        rows, cols = O.shape
                        for r in range(rows):
                            for c in range(cols):
                                # validation_error += -1*target_data[r][c]*np.log(O[r][c])
                                validation_error += -1 * target_data[r][c] * np.log(clipped_output[r][c])
                        # validation_error
                    
                    train_e.append(training_error)
                    validation_e.append(validation_error)
                    epoch.append(e + 1)

                    training_error, validation_error = 0, 0
                # plot the traing_error vs validation error
                plt.plot(epoch, train_e, epoch, validation_e)
                plt.title("Training error -VS- Validatin Error")
                plt.legend(["Training Error", "Validation Error"])
                plt.xlabel("Epochs")
                plt.ylabel("Error")
                plt.show()





            elif self.regression:
                pass

nn = NeuralNetworkTrain()
nn.choose_model(classification = True)
w, b = rwp.read_weights_biases()
inputs = np.zeros((10, 1))
targets = np.zeros((10, 4))
inputs[0][0], inputs[1][0], inputs[2][0], inputs[3][0], inputs[4][0] = 1, 2, 2, 1, 1
inputs[5][0], inputs[6][0], inputs[7][0], inputs[8][0], inputs[9][0] = 1, 2, 2, 2, 1
targets[0][0], targets[1][1], targets[2][2], targets[3][3], targets[4][1] = 1, 1, 1, 1, 1
targets[5][0], targets[6][3], targets[7][2], targets[8][3], targets[9][0] = 1, 1, 1, 1, 1
nn.train_model(inputs, targets, w, b, epochs=100)

