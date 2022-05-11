'''
torch.save(arg, PATH) - use tensors, models or any dict as parameter for saving, 
                        uses python's pickle module to serialize the objects and saves them
                        (so yes, serialized but not readable :))
torch.load(PATH)
model.load_state_dict(arg)

2 options:
    - LAZY way:
    torch.save(model, PATH)
    model = torch.load(PATH) # setting up the saved model
    model.eval()

    - RECOMMENDED way:
        torch.save(model.state_dict(), PATH) # only save the parameters (.state_dict())
        model = Model(*args, **kwargs) # creating the model
        model.load_state_dict(torch.load(PATH)) # load_state_dict the loaded dict
        model.eval()                        

model.eval() MUST BE called to set dropout and batch normalization layers to evaluation mode
before running inference. Failing to do this will yield inconsistent inference results. 
If wanted to resume training, call model.train() to ensure these layers are in training mode.
'''
from pickletools import optimize
import torch
import torch.nn as nn

# Example model
class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model(n_input_features=6)
# train your model...

##### LAZY way #####
FILE = "models/model.pth" # .pth short for pytorch
torch.save(model, FILE)
model = torch.load(FILE)
model.eval()

for param in model.parameters():
    print(param)

#### RECOMMENDED way ######
FILE = "models/model_2.pth"
print(f"model state dict: {model.state_dict()}")
model_2 = torch.save(model.state_dict(), FILE)
loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()

for param in loaded_model.parameters():
    print(param)

# Saving a checkpoint
learning_rate = 0.01
optimizer = torch.optim.SGD(loaded_model.parameters(), lr=learning_rate)
print(f"Optimizer state dict: {optimizer.state_dict()}")

# Manually created checkpoint
checkpoint = {
    "epoch": 90,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict()
}

FILE = "models/checkpoint.pth"
torch.save(checkpoint, FILE)

loaded_checkpoint = torch.load(FILE)
epoch = loaded_checkpoint["epoch"]
model = Model(n_input_features=6)
optimizer = torch.optim.SGD(loaded_model.parameters(), lr=0)
model.load_state_dict(checkpoint["model_state"]) # loading the model state
optimizer.load_state_dict(checkpoint["optimizer_state"]) #loading the optimizer state

print(optimizer.state_dict()) # lr = 0.01 instead of 0 which was initialized when creating the optimizer
'''
# 1) Saving on GPU, loading on CPU ####
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)

device = torch.device("cpu")
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device)) # specify the map location!

# 2) Saving on GPU, loading on GPU ####
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)

model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)

# 3) Saving on CPU, loading on GPU
torch.save(model.state_dict(), PATH)

device = torch.device("cuda")
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # choose whatever GPU device number you want
model.to(device)
# This loads the model to a given GPU device. 
# Next, be sure to call model.to(torch.device('cuda')) to convert the model's parameter tensors to CUDA tensors
'''