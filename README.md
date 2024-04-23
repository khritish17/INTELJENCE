# INTELJENCE Documentation
Overview

## File: neural_architecture.py
### function: build_neural_net(neural_architecture)
> **Description**:
This function builds the neural architecture for a network based on the provided architecture configuration. It generates new weights and biases for each layer of the neural network. If previous weights and biases exist, they will be replaced by the new ones.

⚠️[**WARNING**] This function should be called once to initialize the neural network architecture. Subsequent calls will regenerate new weights and biases, potentially leading to loss of previously learned information.

**Parameters**:

- **'neural_architecture'**: List of integers representing the number of neurons in each layer of the neural network. The first element denotes the input layer, the last element denotes the output layer, and the intermediate elements denote hidden layers.

**Functionality**:

- **Validity Check**: Verifies the validity of the provided neural architecture by ensuring that the number of neurons in each layer is an integer value.
- **Weight Matrix Generation**: Generates weight matrices for each layer using Xavier initialization, where weights are sampled from a uniform distribution with limits calculated based on the number of incoming and outgoing edges (fan-in and fan-out) for each node.
$$weight = U(-limit, limit)$$
$$limit = \sqrt{\frac{6}{fan_{in} + fan_{out}}}$$
> - U: Uniform distribution
> - fan_in: No. of incoming connections to the neuron (No. of neurons in the previous layer)
> - fan_out: No. of outgoing connections from the neuron (No. of neurons in the next layer) 
- **Bias Vector Generation**: Generates bias vectors for each layer using a uniform distribution between -0.1 and 0.1.
- **Metadata Handling**: Creates a metadata folder to store metadata related to the neural network, including the neural architecture configuration.
- **Saving Weights and Biases**: Saves the generated weights and biases into files using the **read_write_parameters.py** module.

## File: read_write_parameters.py
### function: save_weights_biases(weights, biases)
> **Description**:
This function saves the provided weights and biases into separate text files in the specified metadata directory.

**Parameters**:

- **weights** (list of numpy arrays): List containing weight matrices for each layer.
- **biases** (list of numpy arrays): List containing bias vectors for each layer.

**Functionality**:
- **save_weights()**: Converts all weight matrices into a single list and saves them to a text file named 'weights.txt' in the metadata directory.
- **save_biases()**: Saves all bias vectors to a text file named 'biases.txt' in the metadata directory.

### function: read_weights_biases()
> **Description**:
This function reads the saved weights and biases from text files in the metadata directory and returns them.

**Returns**:

- **weights** (list of numpy arrays): List containing weight matrices for each layer.
- **biases** (list of numpy arrays): List containing bias vectors for each layer.

### function: export_weights_biases(destination_path)
> **Description**:
This function copies the saved weights and biases files from the metadata directory to the specified destination directory.

**Parameters**:

- **destination_path**: The path where weights and biases files will be copied.

## File: forward_propagation.py
### function: forward_propagation(inputs)
> **Description**:
This function performs forward propagation through a neural network based on the provided input. It calculates the output of each layer by applying weighted inputs, adding biases, and applying activation functions.

**Parameters**:

- **inputs** (list or numpy array): Input values to the neural network.
**Returns**:
- **output** (list): The output of the neural network after forward propagation.

**Functionality**:

- **weighted_input(inp, weight)**: Calculates the weighted sum of inputs using the provided weights for a layer.
- **leakyReLU(arr)**: Applies the Leaky ReLU activation function element-wise to an array.

Reads the neural architecture, weights, and biases from metadata files using the **read_write_parameters.py** module.

Iterates through each layer of the neural network, performing weighted input summation, adding biases, and applying activation functions (**Leaky ReLU**).
$$Leaky ReLU(x) = max(\alpha \cdot x, x)$$
Returns the final output of the neural network after forward propagation.
> ❗[**Note**]: Ensure that the neural architecture, weights, and biases are correctly configured and saved in the metadata directory before calling this function.



