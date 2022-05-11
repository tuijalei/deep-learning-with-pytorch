import torch

x = torch.randn(3, requires_grad=True) # gradient computing wanted
# From now on, when operations are made with the tensor, 
# pytorch will create a "computational graph"

y = x+2
'''
# x
#   > (+) - y 
# 2
# so got a node with inputs and an output
# backpropagation technigue, gradients can be computed
# first forward pass : calculate the output y
# (then add backward : since gradients needed, pytorch creates a function
# and this function is used in back propagation to get them)
# backward pass : dy/dx
# '''
print(y)

z = y*y*2
print(y)
#z = z.mean() # scalar
print(z)

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward(v) # gradients dz/dx: >> ONLY FOR scalars!! we need to give a vector otherwise
x.grad # gradients: vector Jacobian production!! ( J * v ) in which v = gradient vector

x = torch.randn(3, requires_grad=False) # not gradients
z.backward() # ERROR

# If we do not need gradients, we can use 3 techniques:
# x.requires_grad_(False)
# y = x.detach() | a new tensor with same values
# with torch.no_grad(): y = x + 2

# when ever the backward function is called, gradients for the tensor 
# will be accumulated into the dot fradient attribute so the values will be summed up!
weights = torch.ones(4, requires_grad=True)
for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)

    weights.grad.zero_() # emptying the gradients and will correct the gradients!

optimizer = torch.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad() # before the next iteration!

