'''
Backpropagation (backward propagation) is an important mathematical tool for improving the accuracy of predictions 
in data mining and machine learning. Essentially, backpropagation is an algorithm used to calculate derivatives quickly.

Given an artificial neural network and an error function, the method calculates the gradient of the error function with 
respect to the neural network's weights. It is a generalization of the delta rule for perceptrons to multilayer 
feedforward neural networks.
--------------
1) forward pass: compute loss
2) compute local gradients
3) backward pass: compute dLoss/dWeights using the Chain Rule

Dummy backpropagation

1) Forward pass: x = 1, y = 2, w = 1

    x
     \
      (x) -y_hat- (-) -s- (^2) -loss
     /            /
    w            y

    y_hat = wx = 1*1 = 1
    s = y_hat-y = 1-2 = -2
    loss = s^2 = (-1)^2 = 1

2) Local gradients
    dLoss/dS = d(S^2)/dS = 2s
    dS/dY_hat = d(y_hat-y)/dY_hat = 1
    dY_hat/dw = d(wx)/dw = x

3) Backward pass
    dLoss/dY_hat = dLoss/dS * dS/dY_hat = 2s*1 = -2
    =dLoss/dW = dLoss/dY_hat * dY_hat/dW = -2x = -2

Next let's implement this step by step with PyTorch
'''
import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)

# Forward pass and compute the loss
y_hat = w * x
loss = (y_hat-y)**2
print(loss)

# Backword pass
loss.backward()
print(w.grad) # -2