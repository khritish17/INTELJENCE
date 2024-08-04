# INTELJENCE Documentation
A detailed mathematical explanation of artificial neural networks is provided in the article:
- Title: **The Symphony of Neurons: A Mathematical Exposition on Artificial Neural Networks**
- Author: **Khritish Kumar Behera**
- Link: [https://drive.google.com/file/d/1vnVGqxBUYL0oOZeid1qzORUBc0Z6Xi1m/view?usp=sharing](https://drive.google.com/file/d/1vnVGqxBUYL0oOZeid1qzORUBc0Z6Xi1m/view?usp=sharing)
  
<-Introduction->

## File: Neural_Networks_api.py
### Description:

This code defines a class called **NeuralNetwork** that provides a more user-friendly interface for training and testing multi-class classification models. Here's a breakdown of its functionalities:

1. **Class Initialization**:
- `__init__(self)`:
  - Initializes the object with empty `weights` and `biases` attributes.
  - Attempts to read pre-trained **weights** and **biases** from the `parameters_IO.py` module using `pio.read_weights_biases`. If no weights/biases are found, it gracefully handles the exception using `pass`
2. **Neural Architecture**:
- `neural_architecture(self, architecture)`:
  - Takes a list `architecture` defining the number of neurons in each layer of the neural network.
  - This function calls a function in `neural_architecture.py` to build the network structure.
  - It's not directly used for training but its be helpful for setting up the network internally.
3. **Training**:
- `train(self, inputs, targets, epochs=1000, learning_rate=0.01)`:
  - Takes training data (`inputs` and `targets`), number of epochs (`epochs`), and learning rate (`learning_rate`) as input.
  - Stores the training data in `self.inputs` and `self.targets`.
  - Creates a `Classification_Model` object (`model`) from `classification_model.py` using the provided data and any pre-trained weights/biases (if available).
  - Calls the `model.training` function to train the neural network for the specified number of epochs with the given learning rate.
4. **Testing**:
- `test(self, test_input, test_target)`:
  - Takes testing data point (`test_input`) and its corresponding target (`test_target`) as input.
  - Reads the latest weights and biases using `pio.read_weights_biases`.
  - Creates a `Classification_Model` object (`model`) with empty inputs and targets (weights and biases are loaded from the previous step).
  - Calls the `model.testing` function to evaluate the model on the single test data point.
5. **Saving Model**:
- `save_as(self, location="")`:
  - Takes an optional `location` argument specifying the path to save the model weights and biases.
  - Uses `os.path.abspath` to get the absolute path of the provided location.
  - Calls `pio.export_weights_biases` (from parameters_IO.py) to export the trained weights and biases to the specified location.

## File: parameters_IO.py
### Description:

This Python script provides functions for saving and reading weights and biases associated with the neural network model, along with exporting them to a specified destination. It utilizes the `os`, `shutil`, and `numpy` libraries for file system operations, data manipulation, and numerical computations.

### Functions:
- `save_weights_biases(weights, biases, location="")`
  - Saves the provided weights and biases to the `metadata` directory within the specified `location` (or the current working directory if `location` is empty).
  - **Parameters**:
    - `weights`: A list of NumPy arrays representing the weights for each layer in the neural network.
    - `biases`: A list of NumPy arrays representing the biases for each layer in the neural network.
    - `location` (optional): A string specifying the directory path where the metadata folder will be created (defaults to the current working directory).
  - **Functionality**:
    - Creates the `metadata` directory within the specified `location`.
    - Defines two nested functions:
      - `save_weights()`: Opens the `weights.txt` file in write mode and iterates through each weight array, converting elements to strings, and saving them comma-separated on separate lines.
      - `save_biases()`: Similar to `save_weights()`, but operates on the `biases.txt` file and reshapes each bias array into a single-row NumPy array before saving.
    - Calls both `save_weights()` and `save_biases()` to store the weight and bias data.
- `read_weights_biases(location="")`
  - Reads the previously saved weights and biases from the `metadata` directory within the specified `location` (or the current working directory if location is empty).
  - **Parameters**:
    - `location` (optional): A string specifying the directory path containing the `metadata` folder (defaults to the current working directory).
  - **Returns:**
    - A tuple containing two elements:
      - A list of NumPy arrays representing the weights for each layer.
      - A list of NumPy arrays representing the biases for each layer.
  - **Functionality**:
    - Reads the neural architecture (number of neurons per layer) from the `metadata.txt` file.
    - Defines two nested functions:
      - `read_weights()`: Opens the `weights.txt` file, iterates through lines, converts comma-separated strings back to floats, and reconstructs the weight arrays as NumPy arrays.
      - `read_biases()`: Opens the `biases.txt` file, reads each line, converts comma-separated strings to floats, reshapes them into single-row NumPy arrays, and appends them to the `biases` list.
    - Calls both `read_weights()` and `read_biases()` to retrieve the weight and bias data.
    - Returns the weight and bias lists as a tuple.
- `export_weights_biases(destination_path, source_path="")`
  - Exports the existing weights and biases from the `metadata` directory within the specified `source_path` (or the current working directory if `source_path` is empty) to the `destination_path`.
  - **Parameters**:
    - `destination_path`: A string specifying the absolute path where the weights and biases will be copied.
    - `source_path` (optional): A string specifying the directory path containing the `metadata` folder with the weights and biases (defaults to the current working directory).
  - **Functionality**:
    - Gets the absolute paths for the source and destination directories.
    - Gets the absolute paths for the `weights.txt` and `biases.txt` files within the source metadata directory.
    - Uses `shutil.copy2` to copy both weight and bias files to the destination path.
    - Prints a success message upon completion.
### Example Usage:
**Python**
```
# Assuming weights (list of NumPy arrays) and biases (list of NumPy arrays) are defined

# Save weights and biases to the current working directory
save_weights_biases(weights, biases)

# Read weights and biases from the current working directory
weights, biases = read_weights_biases()

# Export weights and biases to a specific destination path
export_weights_biases("/path/to/destination")
```
## File: neural_architecture.py
### Description:
This Python script defines the `build_neural_net` function, responsible for creating the neural network architecture, including generating and initializing weights and biases. It utilizes functions from the `parameters_IO` module for saving the generated weights and biases.
### Function:
- `build_neural_net(neural_architecture, location="")`
  - Builds the neural network architecture based on the provided neural_architecture (a list of integers representing the number of neurons in each layer).
  - **Parameters**:
    - `neural_architecture`: A list of integers specifying the number of neurons in each layer of the network.
    - `location` (optional): A string specifying the directory path where the network metadata will be stored (defaults to the current working directory).
  - **Functionality**:
    - **Validity Check**:
      - Defines a nested function `validity` that verifies if each element in the `neural_architecture` list is an integer. If not, an error message is printed, and the program exits.
    - **Weight Matrix Generation**:
      - Defines a nested function `generate_weight_matrix` that takes two integers, `dim_x` and `dim_y`, representing the dimensions of the desired weight matrix.
        - Calculates the initialization limit based on the Xavier initialization formula.
        - Generates a random NumPy array using uniform distribution within the calculated `limit`.
    - **Bias Vector Generation**:
      - Defines a nested function `generate_bias_vector` that takes an integer `layer_node_count` representing the number of nodes in a layer.
        - Generates a random NumPy array with dimensions `(1, layer_node_count)` using uniform distribution between -0.1 and 0.1.
    - **Neural Architecture Construction**:
      - Calls the `validity` function to ensure the architecture is valid.
      - Initializes empty lists `weights` and `biases` to store weight matrices and bias vectors for each layer.
      - Iterates through the layers (excluding the last one):
        - Extracts the number of neurons in the current layer and the next layer.
        - Calls `generate_weight_matrix` to create a weight matrix with appropriate dimensions.
        - Appends the generated weight matrix to the `weights` list.
      - Iterates through all layers:
        - Calls `generate_bias_vector` to create a bias vector with the number of nodes in the current layer.
        - Appends the generated bias vector to the `biases` list.
    - **Metadata Management**:
      - Attempts to create a `metadata` directory within the specified `location`.
        - If the directory already exists (potentially leftovers from a previous network), it is deleted using `shutil.rmtree` before creating a new one.
      - Opens `metadata.txt` within the `metadata` directory for writing.
      - Converts the `neural_architecture` list to a comma-separated string and writes it to the file as "nn_architecture:<architecture_string>".
      - Closes the `metadata.txt` file.
    - **Saving Weights and Biases**:
      - Calls the `io.save_weights_biases` function from the `parameters_IO` module to save the generated weights and biases to the `metadata` directory.
**Notes**:
- This code defines a deterministic behavior for generating weights and biases, meaning calling `build_neural_net` multiple times with the same architecture will produce the same weights and biases.
- Consider incorporating options for customizing initialization methods or random seed values for more varied weight and bias generation.

### Example Usage:
**Python**
```
neural_architecture = [1, 3, 4]  # Example architecture with 3 layers: 1, 3, and 4 neurons
build_neural_net(neural_architecture)  # Creates the neural network architecture, weights, and biases
```
## File: predefined_algorithms.py
### Description
This Python script defines various activation functions and their corresponding gradient functions commonly used in neural networks. It also includes the `argmax` function for finding the index of the largest element in an array. Additionally, example code snippets demonstrate usage and visualizations for some functions.

### Functions:
- `sigmoid(array)`: Implements the sigmoid activation function.
  - Takes a NumPy array `array` as input.
  - Applies the element-wise formula `1 / (1 + exp(-array))` to calculate the sigmoid values.
  - Returns a NumPy array containing the sigmoid activations for each element in the input array.
- `relu(array)`: Implements the ReLU (Rectified Linear Unit) activation function.
  - Takes a NumPy array array as input.
  - Applies the element-wise formula `max(array, 0)` to set negative values to zero.
  - Returns a NumPy array containing the ReLU activations for each element in the input array.
- `relu_gradient(array)`: Calculates the gradient of the ReLU activation function.
  - Takes a NumPy array `array` as input, representing the output of the ReLU function.
  - Applies the element-wise formula `1` for elements greater than zero and `0` otherwise.
  - Returns a NumPy array containing the gradients for each element in the input array.
- `sigmoid_gradient(array)`: Calculates the gradient of the sigmoid activation function.
  - Takes a NumPy array `array` as input, representing the output of the sigmoid function.
  - Employs the formula `sigmoid(array) * (1 - sigmoid(array))` for efficient calculation.
  - Returns a NumPy array containing the gradients for each element in the input array.
- `softmax(array)`: Implements the softmax activation function.
  - Takes a NumPy array `array` as input.
  - Applies the softmax formula `exp(array) / sum(exp(array))` with a small epsilon value to avoid division by zero.
  - Normalizes the input array elements to a probability distribution where the sum of all elements equals 1.
  - Returns a NumPy array containing the softmax probabilities for each element in the input array.
- `gradient_softmax(array)`: Calculates the gradient of the softmax activation function.
  - Takes a NumPy array `array` as input, representing the output of the softmax function.
  - Uses the formula `softmax(array) * (1 - softmax(array))` for element-wise calculation.
  - Returns a NumPy array containing the gradients for each element in the input array.
- `argmax(array)`: Finds the index of the largest element in a NumPy array `array`.
  - Sets all elements except the maximum element to zero.
  - Returns a modified version of the input array where only the element with the maximum value is set to 1.

### Example Usage:
**Python**
```
import predefined_algorithms as pa

# Example with ReLU
a = np.array([0.235, -1.25, 0, 0.124])
r = pa.relu(a)
print("ReLU output:", r)
grad_r = pa.relu_gradient(r)
print("ReLU gradient:", grad_r)

# Example with softmax (replace with your desired input array)
array = [...]  # Replace with your input array
softmax_output = pa.softmax(array)
print("Softmax output:", softmax_output)
```

## File: classification_model.py
### Description
This Python script defines a class Classification_Model for building and training a multi-layer neural network for classification tasks. Here's a breakdown of its components and functionalities:

**Class Definition:**
- `Classification_Model(self, inputs, targets, weights, biases)`:
  - Initializes the model with input data (`inputs`), target labels (`targets`), pre-trained weights (`weights`), and biases (`biases`).
  - `inputs`: A 3D NumPy array representing the input data, where each element is a 1D array representing a single data point.
  - `targets`: A 3D NumPy array representing the target labels (one-hot encoded), with the same structure as `inputs`.
  - `weights`: A list of NumPy arrays, where each array represents the weights for a specific layer.
  - `biases`: A list of NumPy arrays, where each array represents the biases for a specific layer.
  - Internal variables:
    - `self.layer_output`: Stores the output for each layer during forward propagation (used in backpropagation).
    - `self.training_inputs`, `self.training_targets`, `self.validation_inputs`, `self.validation_targets`: To store training and validation data.
    - `self.length_training`: Length of the training data.
    - `self.optimal_weights`, `self.optimal_biases`: Stores the weights and biases with the best validation performance.

**Data Splitting:**
- `divide_data(self, training_percentage=80)`:
  - Splits the input data (`inputs` and `targets`) into training and validation sets based on a specified percentage (default: 80% training, 20% validation).

**Forward Propagation:**
- `forward_propagation(self, input)`:
  - Takes a single data point (input) as input.
  - Iterates through each layer, performing matrix multiplication with weights, adding biases, and applying the activation function (ReLU for hidden layers, softmax for the output layer).
  - Stores the output of each layer in self.layer_output.
  - Returns the final output of the network.

**Backpropagation:**
- `backpropagation(self, input, target, learning_rate=0.01)`:
  - Performs backpropagation to update weights and biases.
  - Takes the input data point (`input`), target label (`target`), and learning rate (`learning_rate`) as input.
  - Calculates the error based on the difference between the output and the target.
  - Backpropagates the error through the network, computing gradients for weights and biases using chain rule.
  - Updates weights and biases based on the learning rate and gradients.

**Training:**
- `training(self, epochs, learning_rate=0.01)`:
  - Trains the model for a specified number of epochs (`epochs`) with a given learning rate (`learning_rate`).
  - Iterates through epochs:
    - For each training data point, performs forward propagation and backpropagation.
    - Calculates training error and validation error after each epoch.
    - Keeps track of the epoch with the minimum validation error and stores the corresponding weights and biases as `self.optimal_weights` and `self.optimal_biases`.
    - Plots the training error and validation error vs. epochs.
    - Saves the best weights and biases using the `parameters_IO.save_weights_biases` function.
    - Returns the training error and validation error history.

**Testing:**
- `testing(self, testing_inputs, testing_targets)`:
  - Takes testing data (`testing_inputs` and `testing_targets`) as input.
  - Iterates through the testing data points:
    - Performs forward propagation to get the predicted output.
    - Compares the predicted output with the target label to determine accuracy.
  - Prints the overall number of correct predictions, wrong predictions, and accuracy.

## File: example.py
This Python script demonstrates how to use the classification_model.py class for training and testing a neural network on the MNIST handwritten digit classification task. Here's a breakdown of its functionalities:

**One-Hot Encoding:**
- `one_hot_encode(number)`:
  - Takes an integer (`number`) representing the digit class (0-9).
  - Returns a one-hot encoded representation of the class as a NumPy array.

**Train Model:**
- `train_model(epoch=1000)`:
  - Reads the MNIST training data from a CSV file (`mnist_train.csv`).
  - Iterates through each line in the CSV:
    - Creates a zero-filled NumPy array (`input`) to store the pixel values (normalized to 0-1).
    - Converts the target label (first element in the line) to a one-hot encoded vector using `one_hot_encode`.
    - Splits the remaining line elements into pixel values and stores them in the `input` array.
    - Appends the processed input and target to separate lists (`inputs` and `targets`).
  - Defines the neural network architecture using `na.build_neural_net` (this function exists in `neural_architecture.py`). Here, a simple network with 784 input neurons (28x28 pixels), 10 hidden neurons, and 10 output neurons (for 10 digits) is used.
  - Reads pre-trained weights and biases from the model using `pio.read_weights_biases` (this function exists in `parameters_IO.py`).
  - Initializes a `Classification_Model` object (`model`) with the prepared data and weights/biases.
  - Trains the model for a specified number of epochs (`epoch`) with a learning rate of 0.01 using `model.training`.

**Test Model:**
- `test_model()`
  - Reads the MNIST test data from a CSV file (`mnist_test.csv`).
  - Follows a similar process as `train_model` to prepare the testing inputs and targets.
  - Reuses the pre-trained weights and biases read earlier.
  - Initializes a new `Classification_Model` object (`model`) with empty inputs and targets (weights and biases are loaded from the previous step).
  - Evaluates the model on the testing data using `model.testing`. This calculates the accuracy of the model on unseen data.

**Running the Script:**
- The script defines two functions: `train_model` and `test_model`.
- It calls `train_model(epoch=2000)` to train the model for 2000 epochs.
- After training, it calls `test_model` to evaluate the model's performance on the test data.
> Important Notes:
Modify the file paths ("D:\Codes\Projects\Inteljence\Dataset\mnist_train.csv" and "D:\Codes\Projects\Inteljence\Dataset\mnist_test.csv") to point to the actual location of your MNIST training and testing data.

This example demonstrates a basic workflow for training and testing a neural network for image classification using the provided classification_model.py class.
