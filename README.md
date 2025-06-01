# DATASIM - APSTA - Introduction to Percepton and CNNs

### Claim
This lab was done during my last year of Engineering School at Centrale Nante; the lesson associated to it is named 'APSTA' for 'APprentissage STAtistiques' gave by Professor Diana MATEUS.

## Lab Perceptron

![Screenshot from 2025-06-01 12-46-00](https://github.com/user-attachments/assets/1b48d954-20da-4292-ab3f-d46a788714c2)

### Key Concepts

- **Neural Network Basics**: Understanding the core concepts of neural networks, including weights, biases, and activation functions.
- **Binary Classification**: Classifying input data into one of two categories (e.g., cat vs. non-cat).
- **Data Preprocessing**: Loading, visualizing, and preprocessing image data for use in a neural network.
- **Single Neuron Implementation**: Building a single-neuron model to understand the basics of forward and backward propagation.
- **Cost Function**: Using binary cross-entropy as the cost function to evaluate model performance.
- **Gradient Descent**: Implementing gradient descent for optimizing the model parameters.

### Implementation Steps

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

### Results

- The training process shows a decreasing loss over iterations, indicating that the model is learning.
- The accuracy on the training set is higher than on the testing set, which is expected.
- The single-neuron model provides a basic understanding but is not sufficient for high accuracy in complex tasks.

### Conclusion

This lab provides a foundational understanding of neural networks by implementing a single-neuron model from scratch. It highlights the importance of understanding the underlying mathematics and algorithms before moving on to more complex models and libraries like TensorFlow or PyTorch.

## Lab CNNs

![image](https://github.com/user-attachments/assets/f2634caf-63f8-432c-bce0-102899172d5a)


### Overview

This lab focuses on building and training Convolutional Neural Networks (CNNs) using TensorFlow and Keras. The goal is to classify images into binary categories (e.g., cat vs. non-cat) and explore the impact of different CNN configurations and hyperparameters on model performance.

### Key Concepts

- **Convolutional Neural Networks (CNNs)**: Understanding the architecture and functionality of CNNs, which are particularly effective for image recognition tasks.
- **Data Loading and Preprocessing**: Loading, visualizing, and preprocessing image data for use in a CNN.
- **Model Building**: Constructing a CNN model using Keras and TensorFlow, including convolutional layers, pooling layers, and dense layers.
- **Training and Evaluation**: Training the model using different configurations and evaluating its performance using metrics such as accuracy and ROC curves.
- **Custom Training Loop**: Implementing a custom training loop to gain a deeper understanding of the training process.

### Implementation Steps

1. **Data Loading and Preprocessing**:

- Load and visualize the dataset.
- Normalize and preprocess the image data to prepare it for training.

2. **Building the CNN Model**:

- Construct a CNN model using Keras and TensorFlow.
- Define the architecture with convolutional layers, pooling layers, and a dense output layer with a sigmoid activation function for binary classification.
- Compile the model with binary cross-entropy loss and the Adam optimizer.

3. **Training and Evaluation**:

- Train the model using different hyperparameters such as batch size and number of epochs.
- Evaluate the model's performance on the test set and compute metrics such as accuracy and loss.
- Plot ROC curves to visualize and compare the performance of different CNN configurations.

4. **Custom Training Loop**:

- Implement a custom training loop using TensorFlow's `GradientTape` to manually compute gradients and update model weights.
- Use `tf.data.Dataset` to create an iterable dataset for training.
- Track and visualize training metrics such as loss and accuracy over epochs.

### Results

- The training process shows a decreasing loss and increasing accuracy over epochs, indicating that the model is learning.
- Different batch sizes and epochs impact the model's performance, with larger batch sizes generally leading to faster computation times.
- The ROC curves provide insights into the trade-offs between true positive and false positive rates for different model configurations.

### Conclusion

This lab provides a comprehensive understanding of building, training, and evaluating CNNs using TensorFlow and Keras. It highlights the importance of hyperparameter tuning and the impact of different configurations on model performance. The custom training loop offers a deeper insight into the training process and the optimization of model parameters.
