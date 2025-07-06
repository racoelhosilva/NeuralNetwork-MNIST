# Neural Network Implementation for MNIST 

This repository holds a neural network implementation made from scratch using modern C++. The aim was to experiment with different Neural Network operations, hyperparameters and configurations to better understand how they work. For this, I chose the MNIST dataset as it is a simple and well-known example to focus on.

### 🚧 This repository is still a work in progress! 🚧

## Features

Although the project started out as a simple neural network implementation to be trained on the MNIST dataset, it quickly evolved into something closer to a library or playground. 

Currently, there are multiple configurations and parameters that can be set to experiment with machine learning techniques in different areas including: the initially defined model, the general training setup, the learning methods and the evaluation.

### Architecture

- **Number of Fully Connected Layers**: the network is composed by a customizable amount of layers, allowing for both shallow and deep learning.
- **Configurable Sizes per Layer**: layers in the same network can have different number of units capable of making associations to the adjacent layers.
- **Weights and Biases**: to understand the input data, the neural network balances the associations between units using weights and biases such that triggering the initial input units triggers a chain reaction which propagates until the output layer.
- **Activation Functions**: used per-layer to introduce non-linearities and allow the network to learn the complex data patterns. Currently supports:
  - Rectified Linear Unit (ReLU)
  - Normalized Exponential Function (Softmax)
  - Sigmoid
- **Weight Initialization**: each layer needs randomly generated default values for stability. Some common approaches involve sampling from specific normal or uniform distributions which are more adequate to certain activation functions. Currently supports:
  - LeCun Initialization
  - Glorot/Xavier Initialization
  - He/Kaiming Initialization

### Training

- **Epochs / Iterations**: to allow the model to continuously learn and better recognize the patterns, the fit phase consists of multiple passes (epochs/iterations) through the input data.
- **Input Shuffling**: when the epochs train on the given data always following the same order, there's a risk of "Catastrophic Interference" on the initial data. To prevent this, data can be shuffled before each epoch.
- **Batch Training**: during training each input given to the model will trigger changes to the weights and biases. To accelerate the learning process and provide more stable updates, multiple inputs can be bundled together in a single pass. Currently supports:
  - Individual Training
  - Mini-Batch Training
  - Full-Batch Training

### Evaluation

- **Accuracy and Loss**: whether it is during or after training the model, it's essential to evaluate how well it performs on an untrained data. Currently supports:
  - Loss: measures performance by comparing the error between the prediction and the actual result.
  - Accuracy: measures performance by comparing the number of correctly classified values to the total.
- **Early Stopping with Patience**: if the fit phase is given a validation dataset to compare to, it can calculate the metrics after each epoch. When there are no improvements from the best result for a given number of epochs (patience) it can stop early and stick with the best model.

### Approximation

- **Loss Metric**: at the end of each forward propagation, the network tries to balance itself towards the direction of the input data. The methods used to calculate the direct loss and gradient depend on the metric. Currently supports:
  - Mean Squared Error (MSE)
  - Cross Entropy
- **Optimizer**: after determining the direction to update the weights and biases of each unit (gradient), different techniques can be used to update them. Currently supports:
  - Stochastic Gradient Descent (SGD)
  - SGD with Momentum
  - Root Mean Squared Propagation (RMSProp)
  - Adam (with bias correction)
- **Learning Rate Scheduling**: when updating, the optimizer must how much importance should be given to the calculated gradient (learning rate). When training for multiple epochs, the learning rate should be adjusted (decay) over time. Currently supports:
  - Constant
  - Exponential
  - Inverse Square Root
  - Time Based

### Generalization

- **Regularization Strategies**: one risk about training the model multiple times on the same input is that the final product might not generalize to new and unseen data. One way to prevent this is to add penalties to larger weights. Currently supports:
  - L1/LASSO 
  - L2/Ridge Regression
  - Elastic Net Regularization
- **Weight Decay**: besides the regularization strategies, some optimizers also rely on weight decay which sligthly reduces the importance given to the previous weight values when updating.

### Utilities

- **Matrix Library**: core class made from scratch to handle the math and operation required for Machine Learning.
- **Data Loader**: simple dataset loader for the MNIST binary datasets. Similar structure can be adapted for other datasets as well.
- **Configuration Structure**: all of the configurations mentioned can be tweaked and experimented with by changing the configuration structures on the main source code file.
