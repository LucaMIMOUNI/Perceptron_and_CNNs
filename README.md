# DATASIM - APSTA - Introduction to Percepton and CNNs

# Claim
This lab was done during my last year of Engineering School at Centrale Nante; the lesson associated to it is named 'APSTA' for 'APprentissage STAtistiques' gave by Professor Diana MATEUS.

## Key Concepts

- **Neural Network Basics**: Understanding the core concepts of neural networks, including weights, biases, and activation functions.
- **Binary Classification**: Classifying input data into one of two categories (e.g., cat vs. non-cat).
- **Data Preprocessing**: Loading, visualizing, and preprocessing image data for use in a neural network.
- **Single Neuron Implementation**: Building a single-neuron model to understand the basics of forward and backward propagation.
- **Cost Function**: Using binary cross-entropy as the cost function to evaluate model performance.
- **Gradient Descent**: Implementing gradient descent for optimizing the model parameters.

## Implementation Steps

1. **Data Loading and Preprocessing**:
   - Load and visualize the dataset.
   - Shuffle and split the data into training and testing sets.
   - Normalize and vectorize the image data.

2. **Single Neuron Model**:
   - Initialize parameters (weights and bias).
   - Implement the sigmoid activation function.
   - Perform a forward pass to compute predictions.

3. **Cost Estimation**:
   - Calculate the binary cross-entropy loss to evaluate the model's predictions.

4. **Backpropagation**:
   - Compute gradients of the cost function with respect to the weights and bias.
   - Update the parameters using gradient descent.

5. **Training and Evaluation**:
   - Train the model using gradient descent.
   - Plot the training loss curve to visualize the optimization process.
   - Evaluate the model's accuracy on both training and testing datasets.

## Results

- The training process shows a decreasing loss over iterations, indicating that the model is learning.
- The accuracy on the training set is higher than on the testing set, which is expected.
- The single-neuron model provides a basic understanding but is not sufficient for high accuracy in complex tasks.

## Conclusion

This lab provides a foundational understanding of neural networks by implementing a single-neuron model from scratch. It highlights the importance of understanding the underlying mathematics and algorithms before moving on to more complex models and libraries like TensorFlow or PyTorch.

This format is suitable for a GitHub repository README file, providing a clear and structured overview of your lab.
