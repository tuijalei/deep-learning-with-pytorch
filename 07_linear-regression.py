'''
 PyTorch pipeline
 1) Design model (inputs, outputs, forward pass)
 2) Construct loss and optimizer
 3) Training loop
    - forward pass: compute the prediction
    - backward pass: compute the gradients
    - updating weights
'''

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) Data preparation
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1) # num of rows and num of columns
n_samples, n_features = X.shape # for the model sins input and output dims needed

# 1) Modeling
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# 2) Loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss() # mean squared error
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # stochastic gradient descent

# 3) Training
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass and loss
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    # Backward pass
    loss.backward()
    
    # Update the weights
    optimizer.step()

    # Zero the gradients
    optimizer.zero_grad() # NEVER FORGET

    if(epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch+1}: loss = {loss.item():.4f}")

# Plotting
predicted = model(X).detach().numpy() # new tensor where gradients calculation attribute is false
plt.plot(X_numpy, y_numpy, "ro")
plt.plot(X_numpy, predicted, "b")
plt.show()