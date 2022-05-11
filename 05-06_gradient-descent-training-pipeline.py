'''
 This part is about concrete example how to manually optimize the model using the automatic gradiant computation.
 Here the linear regression algorithm is implemented from a scratch
 > equations to calculate the model prediction 
 > the loss function
 > the numerical computation of the gradients and implementing the formula
 > the gradient decent algorithm to optimize the parameters
 And lastly, we will implement the algorithm using PyTorch packages

 PyTorch pipeline
 1) Design model (inputs, outputs, forward pass)
 2) Construct loss and optimizer
 3) Training loop
    - forward pass: compute the prediction
    - backward pass: compute the gradients
    - updating weights
 '''

from pickletools import optimize
import numpy as np
import torch
import torch.nn as nn

######## NUMPY VERSION (Scratch) ########
# f = w * x
# f = 2 * x
def numpy_model(): 
    print("=== Numpy version of the model (from the scratch) ===")
    X = np.array([1,2,3,4], dtype=np.float32) # training samples
    Y = np.array([2,4,6,8], dtype=np.float32) # testing samples
    w = 0.0 # weights

    # Model prediction
    def forward(x):
        return w*x

    # Loss function (MSE)
    def loss(y, y_predicted):
        return ((y_predicted - y)**2).mean()

    # Gradients (MSE = 1/N * (w*x - y)**2) -> dJ/dw = 1/N 2x (w*x - y)
    def gradient(x, y, y_predicted):
        return np.dot(2*x, y_predicted-y).mean()

    print(f"Prediction before training: f(5) = {forward(5):.3f}")

    # Training
    learning_rate = 0.01
    n_iters = 20

    for epoch in range(n_iters):
        # Prediction = forward pass
        y_pred = forward(X)

        # Loss
        l = loss(Y, y_pred)

        # Gradients
        dw = gradient(X, Y, y_pred)

        # update weights
        w -= learning_rate * dw

        if epoch % 2 == 0:
            print(f"epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}")
        
    print(f"Prediction after training: f(5) = {forward(5):.3f}")

######## TENSOR VERSION (Scratch) ########
def tensor_model(): 
    print("=== Tensor version of the model (from the scratch) ===")
    X = torch.tensor([1,2,3,4], dtype=torch.float32) # training samples
    Y = torch.tensor([2,4,6,8], dtype=torch.float32) # testing samples
    w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True) # weights

    # Model prediction
    def forward(x):
        return w*x

    # Loss function (MSE)
    def loss(y, y_predicted):
        return ((y_predicted - y)**2).mean()

    print(f"Prediction before training: f(5) = {forward(5):.3f}")

    # Training
    learning_rate = 0.01
    n_iters = 100 
    # note: backprogagation aint as exact as the numerical gradient computation
    #       so let's raise iteration number!

    for epoch in range(n_iters):
        # Prediction = forward pass
        y_pred = forward(X)

        # Loss
        l = loss(Y, y_pred)

        # Gradients = backward pass
        l.backward() # dl/dw

        # Remember!
        with torch.no_grad():
            w -= learning_rate * w.grad

        # Zero the gradients!!!
        w.grad.zero_()

        if epoch % 10 == 0:
            print(f"epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}")
        
    print(f"Prediction after training: f(5) = {forward(5):.3f}")

######## PYTORCH VERSION ########
def pytorch_model(): 
    print("=== PyTorch version of the model ===")
    # 2D arrays needed for the nn model
    # num of rows is the num of samples and in each row we have the num of features
    X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32) # training samples
    Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32) # testing samples
    n_samples, n_features = X.shape
    print(f"num of samples: {n_samples} and num of features: {n_features}")

    input_size = n_features
    output_size = n_features
    #model = nn.Linear(input_size, output_size) # one layer linear regression (PyTorch)

    # Dummy example since this is already done before tho
    class LinearRegression(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(LinearRegression, self).__init__()
            # define layers
            self.lin = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return self.lin(x)

    model = LinearRegression(input_size, output_size)
    
    X.test = torch.tensor([5], dtype=torch.float32) # testing tensor
    print(f"Prediction before training: f(5) = {model(X.test).item():.3f}")

    # Training
    learning_rate = 0.01
    n_iters = 100 
    # note: backprogagation aint as exact as the numerical gradient computation
    #       so let's raise iteration number!
    loss = nn.MSELoss() # mean squared error
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # stochastic gradient descent

    for epoch in range(n_iters):
        # Prediction = forward pass
        y_pred = model(X)

        # Loss
        l = loss(Y, y_pred)

        # Gradients = backward pass
        l.backward() # dl/dw

        optimizer.step() # optimizing step

        # Zero the gradients!!!
        optimizer.zero_grad()

        if epoch % 10 == 0:
            [w, b] = model.parameters() # unpacking
            print(f"epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}")
        
    print(f"Prediction after training: f(5) = {model(X.test).item():.3f}")

numpy_model()
tensor_model()
pytorch_model()