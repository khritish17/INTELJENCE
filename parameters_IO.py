import os
import shutil
import numpy as np


# path = os.path.abspath("") +"/metadata"
def save_weights_biases(weights, biases, location = ""):
    path = os.path.abspath(location) +"/metadata"
    # weights are saved in accordance with the interface
    def save_weights():
        with open(path+"/weights.txt", "w") as wf:
            for weight in weights:
                for row in weight:
                    temp = [str(ele) for ele in row]
                    wf.write(",".join(temp))
                    wf.write("\n")
        print("Weights successfully saved !!!")

    # biases are saved in accordance with the layers
    def save_biases():
        with open(path+"/biases.txt", "w") as bf:
            for bias in biases:
                for row in bias:
                    temp = [str(ele) for ele in row]
                    bf.write(",".join(temp))
                    bf.write("\n")
        print("Biases successfully saved !!!")
    
    save_weights()
    save_biases()

def read_weights_biases(location = ""):
    path = os.path.abspath(location) +"/metadata"
    # read the neural architecture from metadata.txt
    neural_architecture = []
    with open(path + "/metadata.txt", "r") as mf:
        temp = mf.readline()
        temp = temp.split(":")
        neural_architecture = temp[-1].split(",")
    neural_architecture = [int(ele) for ele in neural_architecture] 
    
    def read_weights():
        weights = []
        with open(path + "/weights.txt", "r") as wf:
            for no_of_rows in neural_architecture[:-1]:
                weight = []
                for _ in range(no_of_rows):
                    temp = wf.readline()
                    temp = temp.rstrip("\n").split(",")
                    temp = [float(ele) for ele in temp]
                    weight.append(temp)
                
                # converting the weights into np array (2d)
                row, col = len(weight), len(weight[0])
                weight_np = np.zeros(shape = (row, col))
                for r in range(row):
                    for c in range(col):
                        weight_np[r][c] = weight[r][c]
                weights.append(weight_np)
        return weights

    def read_biases():
        biases = []
        with open(path + "/biases.txt", "r") as bf:
            temp = bf.readlines()
            for bias in temp:
                bias = bias.rstrip("\n").split(",")
                bias = [float(ele) for ele in bias]
                bias_np = np.zeros((1, len(bias)))
                for i in range(len(bias)):
                    bias_np[0][i] = bias[i]
                biases.append(bias_np)
        return biases
            
    w = read_weights()
    b = read_biases()
    return w, b

def export_weights_biases(destination_path, source_path = ""):
    path = os.path.abspath(source_path) +"/metadata"
    destination_path = os.path.abspath(destination_path)
    source_path_weights = path + "/weights.txt" 
    source_path_biases = path + "/biases.txt" 
    shutil.copy2(source_path_weights, destination_path)
    shutil.copy2(source_path_biases, destination_path)
    print("Export complete !!!")

# w, b = read_weights_biases()
# print(w)
# print("|\n_\n|")
# print(b)
