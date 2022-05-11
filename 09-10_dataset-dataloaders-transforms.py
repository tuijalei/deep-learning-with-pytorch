'''
Some terminology:
epoch = 1 forward and backward pass of ALL training samples
batch_size = num of training samples in one forward and backward pass
num of iterations = num of passes, each pass using [batch_size] num of samples
e.g. 100 samples and batch_size = 20 ---> 100/20 = 5 iterations for 1 epoch
-----------------
Transforms can be applied to PIL images. rensors, ndarrays, or custom data
during creation of the DataSet

complete list of build-in transforms:
https://pytorch.org/vision/0.9/transforms.html

On Images: CenterCrop, Grayscale, Pad, RandomAffline, Random Crop, 
            RandomHorizontalFlip, RandomRotation, Resize, Scale

On Tensors: LinearTransformation, Normalize, RandomErasing

Conversion: ToPILImage (from tensors or ndarray) ToTensor (from ndarray or PILImage)

Compose multiple Transforms:
    composed = transforms.Compose([Rescale(256), RandomCrop(224)])
'''

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):

    def __init__(self, transform=None):
        # data loading and converting them tensors
        wine_data = np.loadtxt("./data/winedata/wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        
        # Could convert to tensors like: self.x = torch.from_numpy(wine_data[:, 1:])
        # Not done here tho since we can use a transform
        self.x = wine_data[:, 1:]
        self.y = wine_data[:, [0]] # num of samples, 1
        self.n_samples = wine_data.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples

# Transform classes:
# converting a numpy array into a tensor
class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

# Multiplication transform
class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target

# Dataset - tensor
dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))

# Dataset - composed transforms
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

# Training
num_epochs = 2
total_samples = len(dataset)
n_iters = math.ceil(total_samples / 4)
print(total_samples, n_iters)

for epoch in range(num_epochs):

    # Iterating the DataLoader
    for i, (inputs, labels) in enumerate(dataloader):
        # Forward
        if(i+1) % 5 == 0:
            print(f"Epoch: {epoch+1}/{num_epochs}, step {i+1}/{n_iters}, inputs: {inputs.shape}")

# More torchvision datasets
# torchvision.datasets.MNIST() # famous MNIST dataset
# fashion-MNIST, cifar, coco