'''
Activation functions apply a non-linear transformation and  decide whether a neuron should be activated or not.

Typically we have a linear layer in the network that applies a linear transformation
so without activation functions the network is basically just a stacked linear regression model.
> not suited for more complex tasks tho

Conclusion: with a non-linear transformation in between the network can learn better and perform
much complex tasks. After each layer typically activation function is applied.

Popular activation functions: step function, sigmoid, TanH, ReLU, Leaky ReLU, softmax

step function: f(x) = {1 if x >= 0, {0 otherwise | [0, 1]
sigmoid function: f(x) = 1/(1+e^-x) | typically in the last layer of a binary classification problem [0,1]
tanH function: f(x) 2/(1+e^(-2x))-1 | "scaled and shifted sigmoid", hidden layers [-1,1]
reLU function: f(x) = max(0,x) | "if you don't know what to use, use reLU in hidden layers", non-linear, most popular [0, inf]
|---------------------------------------------------------------------------------------------------------------------|
| In the normal reLU the smallest value is 0 so the gradient later in the backpropagation is 0. When the gradient is  |
| 0, these weigths will never be updated and the neurons won't learn anything (dead neurons). ==> leaky reLU to use   |
|---------------------------------------------------------------------------------------------------------------------|
leaky reLU function: f(x) = {x if x >= 0, {a*x otherwise | improved version of reLU, tried to solve the vanishing gradient problem [-inf, inf]
softmax: S(y_i) = e^y_i/sum(e^y_j) | good in last layer in multiclass classification problems, probability as an output [0,1]
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# option 1: nn.modules
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) # linear layer
        self.relu = nn.ReLU() # relu activation function
        self.linear2 = nn.Linear(hidden_size, 1)  # next linear layer
        self.sigmoid = nn.Sigmoid() # next activation function
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

# option 2: activation functions direclty in forward pass
class NeuralNet1(nn.Module):
    # nb! only linear layers init here!
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__() 
        self.linear1 = nn.Linear(input_size, hidden_size) # linear layer
        self.linear2 = nn.Linear(hidden_size, 1)  # next linear layer

        # Easily available:
        # nn.Softmax
        # nn.Tanh
        # nn.LeakyReLU
    
    # activation function directly
    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(x))
        return out