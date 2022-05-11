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
import math 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) Data preparation
bc_data = datasets.load_breast_cancer()
X, y = bc_data.data, bc_data.target
n_samples, n_features = X.shape
print(f"{n_samples} samples and {n_features} features")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Scaling features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Converting to torch tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# Reshape y tensors as column vectors
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1) Modeling
# f = wx * b, sigmoid at the end
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    
    def forward(self, x):
        # Applying first linear layer and then sigmoid function - with a build-in function
        y_pred = torch.sigmoid(self.linear(x)) # value between [0, 1]
        return y_pred

model = LogisticRegression(n_features)

# 2) Loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss() # binary cross entropy
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # stochastic gradient descent

# 3) Training
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass and loss
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    
    # Backward pass
    loss.backward()
    
    # Update the weights
    optimizer.step()

    # Zero the gradients
    optimizer.zero_grad() # NEVER FORGET

    if(epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch+1}: loss = {loss.item():.4f}")

# Evaluating the model
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_classes = y_pred.round()
    acc = y_pred_classes.eq(y_test).sum() / float(y_test.shape[0])
    print(f"Accuracy = {acc:.4f}")

