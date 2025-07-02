# Neural Network Implementation for MNIST 

This repository holds a neural network implementation made from scratch using modern C++. The aim was to experiment with different Neural Network operations, hyperparameters and configurations to better understand how they work. For this, I chose the MNIST dataset as it is a simple and well-known example to focus on.

~~Up until now, the best model was composed of one hidden layer with 80 nodes using ReLU for activation and He for weight initialization, and an output layer (size 10) using softmax and Xavier. The loss function was cross-entropy. The training was done using Stochastic Gradient Descent (SGD) on the entire training dataset through 75 epochs with constant learning rate of 0.01 and no batching, shuffling or L2 regularization. This model achieved a maximum accuracy of **97.98%** over the full test dataset and the entire process took around 56 minutes.~~

Up until now, the best model was composed of two hidden layers with 256 and 32 nodes using ReLU for activation and He for weight initialization, and an output layer (size 10) using softmax and Xavier. The loss function was cross-entropy. The training was done using Stochastic Gradient Descent (SGD) on the entire training dataset through 50 epochs with constant learning rate of 0.01 and no batching, shuffling or L2 regularization. This model achieved a maximum accuracy of **98.1%** over the full test dataset and the entire process took around 2 hours and 10 minutes.

### 🚧 This repository is still a work in progress! 🚧
