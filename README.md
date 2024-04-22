# INTELJENCE Documentation
Overview

## File: neural_architecture.py
### function: build_neural_net(neural_architecture)
**Description**:

This function builds the neural architecture for a network based on the provided architecture configuration. It generates new weights and biases for each layer of the neural network. If previous weights and biases exist, they will be replaced by the new ones.
> ⚠️[WARNING] This function should be called once to initialize the neural network architecture. Subsequent calls will regenerate new weights and biases, potentially leading to loss of previously learned information.

**Parameters**:

- **'neural_architecture'**: List of integers representing the number of neurons in each layer of the neural network. The first element denotes the input layer, the last element denotes the output layer, and the intermediate elements denote hidden layers.

**Functionality**:

- **Validity Check**: Verifies the validity of the provided neural architecture by ensuring that the number of neurons in each layer is an integer value.
- **Weight Matrix Generation**: Generates weight matrices for each layer using Xavier initialization, where weights are sampled from a uniform distribution with limits calculated based on the number of incoming and outgoing edges (fan-in and fan-out) for each node.
- **Bias Vector Generation**: Generates bias vectors for each layer using a uniform distribution between -0.1 and 0.1.
- **Metadata Handling**: Creates a metadata folder to store metadata related to the neural network, including the neural architecture configuration.
- **Saving Weights and Biases**: Saves the generated weights and biases into files using the **read_write_parameters.py** module.
