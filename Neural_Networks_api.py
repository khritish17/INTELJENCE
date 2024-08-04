import classification_model as cm
import neural_architecture as na
import parameters_IO as pio
import os 


class NeuralNetwork:
    def __init__(self) -> None:
        self.weights = None
        self.biases =  None
        try:
            self.weights, self.biases = pio.read_weights_biases()
        except:
            pass
    
    def neural_architecture(self, architecture):
        # build the neural architecture
        na.build_neural_net(architecture)
        
    
    def train(self, inputs, targets, epochs = 1000, learning_rates=0.01):
        self.inputs = inputs
        self.targets = targets
        model = cm.Classification_Model(inputs=self.inputs, targets=self.targets, weights=self.weights, biases=self.biases)
        model.training(epochs= epochs, learning_rate=learning_rates)
    
    def test(self, test_input, test_target):
        self.weights, self.biases = pio.read_weights_biases()
        model = cm.Classification_Model(inputs=[], targets=[], weights=self.weights, biases=self.biases)
        model.testing(testing_inputs= test_input, testing_targets=test_target)

    def save_as(self, location = ""):
        path = os.path.abspath(location)
        pio.export_weights_biases(destination_path=path)
