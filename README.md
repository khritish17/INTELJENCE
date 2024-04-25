# INTELJENCE Documentation
Overview
## Algorithm
### Backpropagation and Regularization
(add the figure of 3x2 layered network)
Equation:

$$O_1 = ReLU(I_1w_1 + I_2w_3 + I_3w_5 +b_1)$$

$$O_2 = ReLU(I_1w_2 + I_2w_4 + I_3w_6 +b_2)$$

$$\begin{bmatrix} O_1 \\ O_2 \end{bmatrix} = ReLU \left(\begin{bmatrix} I_1 \\ I_2 \\ I_3 \end{bmatrix} \times \begin{bmatrix} w_1 \\ w_2 \\\ w_3 \\ w_4 \\\ w_5 \\ w_6\end{bmatrix}\right) + \begin{bmatrix} b_1 \\ b_2 \end{bmatrix}$$

error ($e$) = $T - O$, where $T$ is target/desired value and $O$ is the output of forward propagation.

Squared error ($E$) = $(T - O)^2$
#### Regularization: L1 ($L1_{reg}$)
$E = (T - O)^2$ + $L1_{reg}$

$L1_{reg} = \lambda \cdot sum(|w| + |b|)$ where $\lambda$ is the regularization hyperparameter 

$E = (T - O)^2$ + $\lambda \cdot sum(|w| + |b|)$

For $O_1$, lets say the traget value is $T_1$ hence the Squared error is $E_1$, similarly, for $O_2$, lets say the traget value is $T_2$ hence the Squared error is $E_2$
Equation:

$$E_1 = (T_1 - O_1)^2 + \lambda \cdot sum(|w| + |b|)$$ 

$$E_1 = (T_1 - ReLU(I_1w_1 + I_2w_3 + I_3w_5 +b_1))^2 + \lambda \cdot (|w_1| + |w_2| + |w_3| + |b_1|)$$

similarly,

$$E_2 = (T_2 - ReLU(I_1w_2 + I_2w_4 + I_3w_6 +b_2))^2 + \lambda \cdot (|w_2| + |w_4| + |w_6| + |b_2|)$$

**Weight Update:** $w = w -  L.R \cdot \frac{\partial E}{\partial w}$, where $L.R$ is the learning rate

Hence,

$$E_1 = (T_1 - ReLU(I_1w_1 + I_2w_3 + I_3w_5 +b_1))^2 + \lambda \cdot (|w_1| + |w_2| + |w_3| + |b_1|)$$

> ❗[**Note**]: the regularization term will not contribute to $\frac{\partial E}{\partial w}$

$$\frac{\partial E_1}{\partial w_1} = 2 \cdot (T_1 - ReLU(I_1w_1 + I_2w_3 + I_3w_5 +b_1)) \cdot (- ReLU'(I_1w_1 + I_2w_3 + I_3w_5 +b_1)) \cdot I_1$$

$$Leaky ReLU(x) = \begin{cases} x & \text{if } x \geq 0 \\\ \alpha\cdot x & \text{if } x < 0 \end{cases}$$

$$Leaky ReLU'(x) = \begin{cases} 1 & \text{if } x \geq 0 \\\ \alpha & \text{if } x < 0 \end{cases}$$

> So we can say
>
> $$Leaky ReLU'(x) = \begin{cases} 1 & \text{if } Leaky ReLU(x) \geq 0 \\\ \alpha & \text{if } Leaky ReLU(x) < 0 \end{cases}$$

$$\frac{\partial E_1}{\partial w_1} = 2 \cdot (T_1 - ReLU(I_1w_1 + I_2w_3 + I_3w_5 +b_1)) \cdot (- ReLU'(I_1w_1 + I_2w_3 + I_3w_5 +b_1)) \cdot I_1$$

$$\frac{\partial E_1}{\partial w_1} = 2 \cdot e_1 \cdot (- ReLU'(I_1w_1 + I_2w_3 + I_3w_5 +b_1)) \cdot I_1$$

> Ignore the constant part of $\frac{\partial E_1}{\partial w_1}$

$$\frac{\partial E_1}{\partial w_1} = - e_1 \cdot (ReLU'(I_1w_1 + I_2w_3 + I_3w_5 +b_1)) \cdot I_1$$

$$\frac{\partial E_1}{\partial w_1} = \begin{cases} - e_1 \cdot 1 \cdot I_1 & \text{if } O_1 \geq 0 \\\ -e_1 \cdot \alpha \cdot I_1 & \text{if } O_1 < 0 \end{cases}$$

Matrix we have,

$$Input, I = \begin{bmatrix} I_1 \\ I_2 \\ I_3 \end{bmatrix}_{1\times3}$$

$$ Weight, W = \begin{bmatrix} w_1 \\ w_2 \\\ w_3 \\ w_4 \\\ w_5 \\ w_6 \end{bmatrix}_{3\times2}$$

$$Bias, B = \begin{bmatrix} b_1 \\ b_2 \end{bmatrix}_{1\times2} $$

$$Output, O = \begin{bmatrix} O_1 \\ O_2 \end{bmatrix}_{1\times2} $$

$$Error, e = \begin{bmatrix} e_1 \\ e_2 \end{bmatrix}_{1\times2} $$

$\frac{\partial E}{\partial w}$ can be represented in the matrix form

> $E_1$ was contributed by $w_1$, $w_3$, $w_5$ and $E_2$ was contributed by $w_2$, $w_4$, $w_6$, hence

$$ \frac{\partial E}{\partial w} = \begin{bmatrix} \frac{\partial E_1}{\partial w_1} \\ \frac{\partial E_2}{\partial w_2} \\\ \frac{\partial E_1}{\partial w_3} \\ \frac{\partial E_2}{\partial w_4} \\\ \frac{\partial E_1}{\partial w_5} \\ \frac{\partial E_1}{\partial w_6} \end{bmatrix}_{3\times2}$$

$$ \frac{\partial E}{\partial w} = -1 \begin{bmatrix} e_1 \cdot f_1 \cdot I_1 & e_2 \cdot f_2 \cdot I_1 \\\ e_1 \cdot f_1 \cdot I_2 & e_2 \cdot f_2 \cdot I_2 \\\ e_1 \cdot f_1 \cdot I_3 & e_2 \cdot f_2 \cdot I_3 \end{bmatrix}_{3\times2} $$

where, $f_1 = f(O_1)$ and $f_2 = f(O_2)$

$$f(O) = \begin{cases} 1 & \text{if } O \geq 0 \\\ \alpha & \text{if } O < 0 \end{cases}$$

$$\frac{\partial E}{\partial w} = \begin{bmatrix} I_1 \\\ I_2 \\\ I_3 \end{bmatrix} \times \left(-1 \cdot \begin{bmatrix} e_1 & e_2 \end{bmatrix} \cdot \begin{bmatrix} f_1 & f_2 \end{bmatrix} \right)$$

$$F = \begin{bmatrix} f_1 & f_2 \end{bmatrix}$$

$$\frac{\partial E}{\partial w} = I^T\times (-1\cdot e \cdot F)$$

> ⚠️ ' $\times$ ' represents matrix multiplication, and ' $\cdot$ ' represents element wise multiplication (both scalar and matrix)

### weight Update
$$W = W - LR \cdot \frac{\partial E}{\partial w}$$

### Bias Update

$$B = B - LR\cdot \frac{\partial E}{\partial b}$$

$$\frac{\partial E}{\partial b} = -1\cdot e \cdot F$$

> Observation: $\frac{\partial E}{\partial w} = I^T\times \frac{\partial E}{\partial b}$

### Error propagation for the next 'previous' layer

$$e_{new} = e \times W^T + regularization$$

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



