from pkg_resources import require
import torch
import numpy as np

# Empty tensors
v_1D = torch.empty(3) # 1D vector with 3 elements
v_2D = torch.empty(2, 3) # 2D vector
v_3D = torch.empty(2, 2, 3) # 3D
print("{} \n {} \n {}".format(v_1D, v_2D, v_3D))

# Tensors with random values
x = torch.rand(2,2)
only_zeros = torch.zeros(2,2) # only 0s
only_ones = torch.ones(2,2) # only 1s
print("{} \n cd {} \n {}".format(x, only_zeros, only_ones))

# Spesific typed tensors
int_tensor = torch.ones(2,2, dtype=torch.float64) # float typed
double_tensor = torch.ones(2,2, dtype=torch.double) # float typed
print(int_tensor.size()) # size of the tensor

# Constract tensors from data - # list of elements
const_tensor = torch.tensor([2.4, 1.0]) 

### Basic operations ###
x = torch.rand(2,2)
y = torch.rand(2,2)
print("x: {} \n y: {}".format(x, y))
z = x + y # element-wise addition: add up each entries
print("Sum of x and y: {}".format(z))
y.add_(x) # note: every function with _ does the implace operation
print(y) # modify y and add all of the element of x in y

z = torch.add(x,y)
z = torch.sub(x, y)
z = torch.mul(x,y)
z = torch.div(x, y)

# Slicing operations
x = torch.rand(5,3)
print(x[:, 0])
print(x[1, :])
# if only one element in tensor, .item() can be used

# Reshaping
x = torch.rand(4,4)
print(x)
y = x.view(16) # 1d vector
print(y)
y = x.view(-1, 8) # [2, 8] size
print(y)

# Converting numpy to tensor and wise versa
a = torch.ones(5)
print("a before addition: {}".format(a))
b = a.numpy() # converting a numpy array from a tensor
print("b: {}: {}".format(type(b), b))

# note!!! if tensor is in the CPU, both object share the same memory location
# so if we change one, we change the other too!!! For example:
a.add_(1)
print("a after addition: {}".format(a))
print("b after addition in a: {} !!!".format(b))

a = np.ones(5)
print(a)
b = torch.from_numpy(a) # converting a tensor from a numpy array
# Same happen here: if we modify the numpy array, it modifies the tensor too!!

# Tensors can be used in GPU only with CUDA
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device) # putting tensors into the GPU
    y = torch.ones(5)
    y = y.to(device) # moving tensor
    z = x + y
    z.numpy() # ERROR since numpy can only handle CPU tensors!!
    z = z.to("cpu") # back to the CPU

x = torch.ones(5, requires_grad=True) # by default False
# pytorch needs to compute gradients for the tensor later at the optimization steps
# whenever we have a variable which needs to be optimized, gradients are needed
