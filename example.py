import numpy as np
import classification_model as cm
import neural_architecture as na
import parameters_IO as pio
from matplotlib import pyplot as plt
def one_hot_encode(number):
    encodedation = {0 : np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 1 : np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
                    2 : np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]), 3 : np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
                    4 : np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), 5 : np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
                    6 : np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]), 7 : np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
                    8 : np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]), 9 : np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])}
    return encodedation[number]

train_location = "D:\Codes\Projects\Inteljence\Dataset\mnist_train.csv"
test_location = "D:\Codes\Projects\Inteljence\Dataset\mnist_test.csv"
def train_model(epoch = 1000):
    # read the csv file 'mnist_train.csv'
    mnist_train_file = open(train_location) # change the location of training data
    lines = mnist_train_file.readlines()
    total_length = len(lines[0].split(",")) # 785: 1(target) + 28 x 28(= 784 pixel values)
    inputs, targets = [],[]
    print("Reading training data")
    count = 0
    for line in lines:
        input = np.zeros((1, 784))
        line = line.split(",")
        one_hot = one_hot_encode(int(line[0]))
        target = np.zeros((1, 10))
        for i in range(10):
            target[0][i] = one_hot[i]
        for i, ele in enumerate(line[1:]):
            input[0][i] = int(ele)/255
        inputs.append(input)
        targets.append(target)
        count += 1
        percentage = (count/len(lines))*100
        if percentage % 10 == 0:
            print(f"{percentage}% reading complete")
    print("Reading training data complete")
    mnist_train_file.close()

    # na.build_neural_net([784, 10, 10])
    w, b = pio.read_weights_biases()
    model = cm.Classification_Model(inputs= inputs, targets=targets, weights=w, biases=b)

    model.training(epoch, 0.01)

def test_model():
    # read the csv file 'mnist_test.csv'
    w, b = pio.read_weights_biases()
    model = cm.Classification_Model(inputs=[], targets=[], weights=w, biases=b)

    mnist_test_file = open(test_location)
    lines = mnist_test_file.readlines()
    total_length = len(lines[0].split(",")) # 785: 1(target) + 28 x 28(= 784 pixel values)
    testing_inputs, testing_targets = [],[]
    print("Reading testing data")
    count = 0
    for line in lines:
        input = np.zeros((1, 784))
        line = line.split(",")
        one_hot = one_hot_encode(int(line[0]))
        target = np.zeros((1, 10))
        for i in range(10):
            target[0][i] = one_hot[i]
        for i, ele in enumerate(line[1:]):
            input[0][i] = int(ele)/255
        testing_inputs.append(input)
        testing_targets.append(target)
        count += 1
        percentage = (count/len(lines))*100
        if percentage % 10 == 0:
            print(f"{percentage}% reading complete")
    print("Reading testing data complete")
    mnist_test_file.close()

    model.testing(testing_inputs=testing_inputs, testing_targets=testing_targets)

# train_model(epoch=2000)
test_model()

