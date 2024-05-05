# INTELJENCE Documentation
Overview

## Neural Network: Multi-class classification model
### Activation Functions (used in Multi-class classification model)
#### 1. Sigmoid Activation (used in hidden layer neurons)

$$Sigmoid(x) = \frac{1}{1 + e^{-x}}$$

#### 2. Softmax Activation (used in output layer neurons)


$$Softmax(x_i) = \frac{e^{x_i}}{e^{x_1} + e^{x_2} +...+e^{x_n}}$$

### Forward Propagation
(add the figure 3 x 2 interface)

$O$ = $A$(sum of $weighted$ $inputs$ + $bias$), where $A$ is the activation function.

$$O = A \left(\sum i\cdot w + b \right)$$

**Matrix**

$$Input, I = \begin{bmatrix} i_1 & i_2 & i_3 \end{bmatrix}$$

$$ Weight, W = \begin{bmatrix} w_1 & w_2 \\\ w_3 & w_4 \\\ w_5 & w_6 \end{bmatrix}$$

$$Bias, B = \begin{bmatrix} b_1 & b_2 \end{bmatrix}$$

$$Output, O = \begin{bmatrix} O_1 & O_2 \end{bmatrix}$$

**Summation of Weighted Input**

$$\sum i\cdot w = \begin{bmatrix} i_1 & i_2 & i_3 \end{bmatrix} \times \begin{bmatrix} w_1 & w_2 \\\ w_3 & w_4 \\\ w_5 & w_6 \end{bmatrix} = \begin{bmatrix}I_1w_1 + I_2w_3 + I_3w_5 & I_1w_2 + I_2w_4 + I_3w_6\end{bmatrix} = I \times W$$

**Adding Bias Term**

$$\sum (i\cdot w) + b = \begin{bmatrix}I_1w_1 + I_2w_3 + I_3w_5 & I_1w_2 + I_2w_4 + I_3w_6\end{bmatrix} + \begin{bmatrix} b_1 & b_2 \end{bmatrix} = \begin{bmatrix}I_1w_1 + I_2w_3 + I_3w_5 + b_1 & I_1w_2 + I_2w_4 + I_3w_6 + b_2\end{bmatrix}$$

$$\sum (i\cdot w) + b = I \times W + B = W_{IB}$$

For **Sigmoid Activation** (in hidden layers)

$$O = Sigmoid \left(\sum i\cdot w + b \right) = Sigmoid \left(W_{IB} \right)$$

> $$Sigmoid(x) = \frac{1}{1 + e^{-x}}$$

> $$ W_{IB} = \begin{bmatrix}I_1w_1 + I_2w_3 + I_3w_5 + b_1 & I_1w_2 + I_2w_4 + I_3w_6 + b_2\end{bmatrix} = \begin{bmatrix}W_{IB1} & W_{IB2}\end{bmatrix}$$

$$O = \begin{bmatrix} \frac{1}{1 + e^{-W_{IB1}}} & \frac{1}{1 + e^{-W_{IB2}}}\end{bmatrix}$$


For **Softmax Activation** (in output layer)

$$O = Softmax \left(\sum i\cdot w + b \right)$$

> $$Softmax(x_i) = \frac{e^{x_i}}{e^{x_1} + e^{x_2} +...+e^{x_n}}$$

> $$ W_{IB} = \begin{bmatrix}I_1w_1 + I_2w_3 + I_3w_5 + b_1 & I_1w_2 + I_2w_4 + I_3w_6 + b_2\end{bmatrix} = \begin{bmatrix}W_{IB1} & W_{IB2}\end{bmatrix}$$

$$O = \begin{bmatrix} \frac{e^{W_{IB1}}}{e^{W_{IB1}} + e^{W_{IB2}}} & \frac{e^{W_{IB2}}}{e^{W_{IB1}} + e^{W_{IB2}}}\end{bmatrix}$$

### Backpropagation
In order to optimize the weights and biases, we will use gradient descent on both

$$W_{updated} = W - LR \cdot \frac{\partial E}{\partial W}$$

$$B_{updated} = B - LR \cdot \frac{\partial E}{\partial B}$$

> LR is the Learning Rate, this value is set to 0.001 by default

(3x2 figure)
#### Hidden layer - Output Layer Interface
Since the output layer uses the Softmax, the appropriate Error/Loss function is **Cross Entropy Loss**, E

$$E = -\sum (t_i \cdot log(p_i))$$

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
Returns the final> ❗[**Note**]: output of the neural network after forward propagation.
 Ensure that the neural architecture, weights, and biases are correctly configured and saved in the metadata directory before calling this function.



