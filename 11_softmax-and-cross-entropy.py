'''
Softmax: applies the exponential function to each element and normalizes it by dividing it by
         the sum of all of the exponential so basicly squashes the output to be between [0,1]

         Linear -> <scores/logits> -> Softmax -> <probabilities>

Cross-Entropy loss function: usually compined with the softmax function; measures the performance
                             of the classification model whose output is a probability between [0, 1]
                             and can be used in multi-class problems; the loss increases as the predicted 
                             probability diverges from the actual label


                             labels must be one hot encoded!
'''

import torch
import torch.nn as nn
import numpy as np

#### Softmax ####
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Numpy
arr = np.array([2.0, 1.0, 0.1])
outputs = softmax(arr)
print(f"Softmax numpy: {outputs}")

# PyTorch
tensor = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(tensor, dim=0) #computes along the first axis
print(f"Spftmax PyTorch: {outputs}")

#### Cross-entropy ####
def cross_entropy(actual, predicted):
    return -np.sum(actual * np.log(predicted))

# Numpy
# y must be one hot encoded
# if class 0: [1 0 0]
# if class 1: [0 1 0]
# if class 2: [0 0 1]
Y = np.array([1, 0, 0])
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])

l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print("Numpy:")
print(f"Loss1 numpy: {l1:.4f}") # 0.3567
print(f"Loss2 numpy: {l2:.4f}") # 2.3026

# PyTorch
loss = nn.CrossEntropyLoss() # nn.LogSoftmax + nn.NLLLoss (negative log likelihood loss)
# do not use softmax!!

# 1 sample case
Y = torch.tensor([0]) 
# nsamples * nclasses = 1*3 = 3
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]]) # nb! raw values (no softmax!)
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print("PyTorch:")
print("---- One sample case: ----")
print(f"Loss1 numpy: {l1.item():.4f}") # 0.3567
print(f"Loss2 numpy: {l2.item():.4f}") # 2.3026

_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(predictions1, predictions2) # tensor([0]) tensor([1]) nb: indexes

# 3 sample case
Y_2 = torch.tensor([2, 0, 1])
# nsamples * nclasses = 3*3 = 9
Y_pred_good2 = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]]) # nb! raw values (no softmax!)
Y_pred_bad2 = torch.tensor([[2.1, 2.0, 0.1], [0.1, 1.0, 2.1], [0.1, 3.0, 0.1]])

l12 = loss(Y_pred_good2, Y_2)
l22 = loss(Y_pred_bad2, Y_2)

print("---- Three sample case: ----")
print(f"Loss1 numpy: {l12.item():.4f}") # 0.3567
print(f"Loss2 numpy: {l22.item():.4f}") # 2.3026

_, predictions1 = torch.max(Y_pred_good2, 1)
_, predictions2 = torch.max(Y_pred_bad2, 1)
print(predictions1, predictions2) # tensor([0]) tensor([1]) nb: indexes

#### Neural networks in binary and multiclass problems ####
# The only difference is in the forward pass function:
#   In binary problems sigmoid is needed BUT
#   in multiclass problem softmax IS NOT NEEDED

# Binary classification - "Is it a dog?"
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)  
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # sigmoid at the end
        y_pred = torch.sigmoid(out)
        return y_pred

model = NeuralNet1(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss()

# Multi-class classification - "Which animal is it?"
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax at the end
        return out

model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss()  # applies Softmax