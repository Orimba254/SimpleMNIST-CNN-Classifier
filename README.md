# SimpleMNIST-CNN-Classifier

This code is a C++ program that implements a simple Convolutional Neural Network (CNN) from scratch, designed to work on the MNIST dataset, a collection of handwritten digits commonly used for training and testing in the field of machine learning. The program is structured to perform the following tasks:

Class Definition (SimpleCNN): Defines a simple CNN with basic functionalities, including the initialization of weights and biases, a forward pass, a backward pass for learning, and a prediction method. The CNN includes a convolutional layer followed by a fully connected (FC) layer.

Data Loading Functions:

readMNISTImages: Reads image data from a given file path, expected to be in the IDX file format used by the MNIST dataset. It normalizes the pixel values to a range of [0, 1].
readMNISTLabels: Reads label data (the correct digit for each image) from a given file path, also in the IDX format.
Main Functionality:

Initializes the CNN.
Loads training and test data (images and labels) from specified paths.
Trains the CNN using the training data. Training involves performing a forward pass to compute predictions, comparing these predictions against the actual labels, and then adjusting the model's weights and biases based on the errors (backward pass).
Evaluates the trained model on test data to calculate its accuracy, i.e., the percentage of test images for which the model correctly predicts the digit.
Key components and operations in the CNN include:

Convolution: Applies a set of filters (convWeights) to the input images to produce a set of feature maps, followed by adding biases (convBiases).
Activation Function: Uses the ReLU (Rectified Linear Unit) function, implemented as max(0.0, x) for each neuron's output, to introduce non-linearity.
Flattening: Converts the 2D feature maps into a 1D vector to feed into the fully connected layer.
Fully Connected Layer: Processes the flattened vector through a layer of neurons (fcWeights and fcBiases), applying the ReLU activation function again.
Backward Pass (Learning): Adjusts the model weights and biases using a simple form of backpropagation, based on the difference between the predicted and actual labels (error gradient) and a predefined learning rate.
The program is a simplified demonstration of how a CNN can be implemented and trained to recognize handwritten digits, showcasing fundamental concepts like convolution, activation, and backpropagation without relying on any deep learning libraries.
