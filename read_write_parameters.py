import os
import shutil


path = os.path.abspath("") +"/metadata"
def save_weights_biases(weights, biases):
    # weights are saved in accordance with the interface
    def save_weights():
        # convert all weights matrix to one list 
        data = []
        for weight in weights:
            for row in weight:
                data.append(row)
        with open(path+"/weights.txt", "w") as wf:
            for row in data:
                temp = [str(ele) for ele in row]
                wf.write(",".join(temp))
                wf.write("\n")
        print("Weights successfully saved !!!")
    
    # biases are saved in accordance with the layers
    def save_biases():
        with open(path+"/biases.txt", "w") as bf:
            for bias in biases:
                temp = [str(ele) for ele in bias]
                bf.write(",".join(temp))
                bf.write("\n")
        print("Biases successfully saved !!!")
    
    save_weights()
    save_biases()

def read_weights_biases():
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
                weights.append(weight)
        return weights

    def read_biases():
        biases = []
        with open(path + "/biases.txt", "r") as bf:
            temp = bf.readlines()
            for bias in temp:
                bias = bias.rstrip("\n").split(",")
                bias = [float(ele) for ele in bias]
                biases.append(bias)
        return biases
            
    w = read_weights()
    b = read_biases()
    return w, b

def export_weights_biases(destination_path):
    destination_path = os.path.abspath(destination_path)
    source_path_weights = path + "/weights.txt" 
    source_path_biases = path + "/biases.txt" 
    shutil.copy2(source_path_weights, destination_path)
    shutil.copy2(source_path_biases, destination_path)
    print("Export complete !!!")
