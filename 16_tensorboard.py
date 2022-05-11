'''
TensorBoard: TensorFlow's visualization toolkit
TensorBoard provides the visualization and tooling needed for machine learning experimentation:

    Tracking and visualizing metrics such as loss and accuracy
    Visualizing the model graph (ops and layers)
    Viewing histograms of weights, biases, or other tensors as they change over time
    Projecting embeddings to a lower dimensional space
    Displaying images, text, and audio data
    Profiling TensorFlow programs
    And much more

'''
# Let's use the code from the part 13 about feed forward nets

# to solve the following error
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

##################################################
##### TENSOR BOARD ####
import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/mnist")
##################################################

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda if GPU support, CPU otherwise

# Hyper parameters
input_size = 784 # images will have the size of 28x28
hidden_size = 100 # can use different sizes
num_classes = 10 # 10 different classes in the dataset
num_epoch = 2
batch_size = 100
learning_rate = 0.001

# Data - MNIST
train_data = torchvision.datasets.MNIST(
    root = "./data", # root directory of dataset where MNIST/raw/train-images-idx3-ubyte and MNIST/raw/t10k-images-idx3-ubyte exists.
    train = True, # If True, creates dataset from train-images-idx3-ubyte, otherwise from t10k-images-idx3-ubyte.
    transform = transforms.ToTensor(),  # A function/transform that takes in an PIL image and returns a transformed version.
    download = True # If True, downloads the dataset from the internet and puts it in root directory. Otherwise, it is not downloaded again.
)

test_data = torchvision.datasets.MNIST(
    root = "./data", 
    train = False, # If True, creates dataset from train-images-idx3-ubyte, otherwise from t10k-images-idx3-ubyte.
    transform = transforms.ToTensor()
)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape) 
# torch.Size([100, 1, 28, 28]) | 100 samples, 1 channel, 28, 28 image array
# torch.Size([100]) | for each image we have a class value

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap="gray") # samples [0] to get the channel 1

#plt.show()
##################################################
# TENSORBOARD: image grid
img_grid = torchvision.utils.make_grid(samples)
writer.add_image("MNIST images", img_grid)
##################################################

# Model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # we don't want to use the activation function here since we will use cross entropy
        return out

model = NeuralNet(input_size, hidden_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss() # applying softmax for us!!
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

##################################################
# TENSORBOARD: Model graph
writer.add_graph(model, samples.reshape(-1, 28*28))
##################################################

# Training loop
n_total_steps = len(train_loader)

running_loss = 0.0
running_correct = 0
for epoch in range(num_epoch):
    for batch_i, (images, labels) in enumerate(train_loader):
        # Reshaping images first
        # num of batches 100, image_size 784
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predictions = torch.max(outputs, 1)
        running_correct += (predictions == labels).sum().item()
        if (batch_i+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epoch}, step {batch_i+1}/{n_total_steps}: loss = {loss.item():.4f}")
            ##################################################
            # TENSORBOARD: statistics
            writer.add_scalar("Training loss", running_loss / 100, epoch * n_total_steps + batch_i)
            writer.add_scalar("Accuracy", running_correct / 100 , epoch * n_total_steps + batch_i)
            running_loss = 0.0
            running_correct = 0
            ##################################################

# Testing and evaluation

class_labels = []
class_preds = []
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        # value, index (class label)
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item() # for each correct prediction adding +1

        class_labels.append(predictions)
        # linear layer at the end of neural net but we need the softmax to get the probabilities
        class_batch_probs = [F.softmax(output, dim=0) for output in outputs]
        class_preds.append(class_batch_probs)

    # Converting to tensors
    # stack concatenates tensors along a new dimension
    # cat concatenates tensors in the given dimension
    class_preds = torch.cat([torch.stack(batch) for batch in class_preds])
    class_labels = torch.cat(class_labels)

    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy = {acc}")
    ##################################################
    # TENSORBOARD: Precision recall curve for each class
    classes = range(10)
    for i in classes:
        labels_i = class_labels == i
        preds_i = class_preds[:, i] # classes from the each column
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
        writer.close() # All the outputs are being flushed
    ##################################################
    #sys.exit() # Not to read the whole code

