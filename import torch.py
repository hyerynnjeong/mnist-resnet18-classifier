# %%
import torch
from torchvision import datasets, transforms

## DEFINE DATA TRANSFORMATIONS

# transform: convert PIL image → tensor and normalize
transform = transforms.Compose([
    transforms.ToTensor(), #converts PIL image --> PyTorch tensor [1,28,28]
    transforms.Normalize((0.1307,), (0.3081,)) 
    #stadard MNIST normalization 
    #(mean = 0.1307, std = 0.3081)
    #improves training stability
])

# %%

## LOAD THE MNIST DATASET

# train dataset
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# test dataset
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)
#Loads MNIST:
#       60,000 training images
#       10,000 test images
#Each item = (image_tensor, label)

# %%
x, y= train_dataset[2] # x: image tensor , y: digit label
x
x.shape #shows shape --> [1,28,28] : 1 channel (grayscale), 28×28 pixels
y

# x = torch.rand(5, 3)
# print(x)

# %%

## INSPECT / VISUALIZE A SAMPLE

import matplotlib.pyplot as plt
x_np = x.numpy()
x_np.shape
x_np = x_np.squeeze(0) #index 0: channel/ 1: height/ 2: width
x_np.shape
# remove the channel dimension as numpy expects 2D array
# [1,28,28] into (28,28)
    # 1 channel for Greyscale (brightness) only
    # 3 channel for red, green, blue layer --> colored
 
plt.matshow(x_np) #colored bc a visual mapping by matplotlib, not real color channels
plt.matshow(x_np, cmap='grey')
plt.matshow(x_np, cmap='grey_r') #reverse 
# Displays the digit using different colormaps

# %%

## CHECK FILE DIRECTORY

import os
os.getcwd() # = get current working directory
os.listdir() # Shows files present
# Useful for debugging dataset download location.

# %%

## CREATE DATALOADERS

from torch.utils.data import DataLoader

# Tools that loads batches of training data from the dataset, optionally shuffled
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) 
'''
- load 64 images at once (batch: small group of samples)
- randomize order every epoch(one full pass through dataset = batch1 + batch2 + ...)
- loads training data for learning
'''

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
'''
testing dont need random order; just want to measure performance
loads test data for evaluating the model (check accuracy)
'''

# %%
# np.random.randint()

# %%

## DEFINE THE NEURAL NETWORK

import torch.nn as nn # building neural network layers
import torchvision.models as models # prebuilt models like ResNet, VGG, etc

# loads the ResNet-18 architecture
network = models.resnet18(pretrained=False) # 18 learnable layers
'''
resnet-18: Convolutional Neural Network (CNN) with 18 layers
-> originally designed for ImageNet classification
-> different from MNIST
-> need to modify the network
'''

# change input channel 3 (ImageNet images are RGB) → 1 (MNIST)
network.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
'''
- Conv2d scans the image with filters (kernels) to detect patterns like edges, curves, or shapes
- 1: number of input channel
- 64: number of filters (output channels) 
       - each filter detects a different features such as:
            vertical edges/ horizontal edges/ curves/ corners/ digit strokes
- kernel_size=7: size of the filter
        - 7×7 pixels at a time to detect patterns
- stride=2: controls how far the filter moves each step
        - move 2 pixels at a time
- padding=3: adds extra pixels around the border of the image
        - [28 × 28] --> [34 × 34]
        - keep edge information and control output size
- bias=False: removes the bias term
        - normally layers compute [output = weight * input + bias]
        - it's common bc batch normalization layers handle the bias effect

now the model accepts [batch, 1, 28, 28]
'''
network
# change output 1000 → 10 classes
network.fc = nn.Linear(network.fc.in_features, 10)

network.fc.in_features
network.fc.out_features
network.fc.weight.shape
network.fc.bias.shape
list(network.fc.named_parameters())

network.fc.weight.grad

'''
[batch_size, 10]
'''



# %%

##TRAIN THE MODEL HERE

import torch.optim as optim # PyTorch keeps optimization algorithms in torch.optim
'''
optimizers:
SGD	     |  simple gradient descent
Adam	 |  adaptive learning rate 
RMSProp	 |  adaptive method
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = network.to(device)
# Moves model to GPU if available (faster training)

criterion = nn.CrossEntropyLoss() # defines how the model measures error
optimizer = optim.Adam(network.parameters(), lr=0.001) # The optimizer updates the model weights
'''
network.parameters(): weights of the neural network
lr=0.001: learning rate (controls how big each update step is)
'''

epochs = 3 # model sees the full dataset 3 times
'''
Epoch = one full pass through the entire dataset
'''

for epoch in range(epochs):
    network.train() #This tells PyTorch: The model is currently training
    running_loss = 0.0 #average loss
    correct = 0 #correct predictions
    total = 0 #number of samples
    #to compute training accuracy

    for images, labels in train_loader: # gets batches from the DataLoader
        images = images.to(device)
        labels = labels.to(device)
        # if the model is on GPU, the data must also be on GPU.

        logits = network(images) # The images go through the neural network
                                # ex) [0.01, 0.92, 0.03, ...]
        loss = criterion(logits, labels)
        # measures how wrong the predictions are
        # Lower loss = better predictions

        optimizer.zero_grad() #Gradients accumulate in PyTorch. So before each step we reset them
        loss.backward() #computes gradients for every weight in the network --> backpropagation algorithm
                        # how much changing each weight will change the loss
        optimizer.step() #uses the gradients to update the model weights. So the model improves slightly
        '''
        loss measures error
        backpropagation finds which weights caused the error
        optimizer updates those weights
        '''
        running_loss += loss.item() 
        # loss.item() converts the tensor loss to a number
        # accumulate it to compute average loss later

        # _, preds = torch.max(logits, 1)
        preds = torch.argmax(logits, 1)
        '''
        The model outputs 10 values (logits) -> [-10, -11, 3, ...]
        You can change that into 10 probabilities (y_score) -> [0.01, 0.02, 0.91, ...]
        argmax selects the highest probability class (y_pred) -> digit = 2
        '''

        correct += (preds == labels).sum().item() # count correct predictions
        total += labels.size(0)

    train_acc = 100 * correct / total # compute training accuracy
    print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")


# %%

##TEST THE MODEL

# sklearn.metrics.classification_report
from sklearn.metrics import classification_report 
# generates a summary of classification performance

network.eval() # The model is now in testing mode - Because some NN layers (like Batch norm) have train/test modes
            # important for correct evaluation

y_true = []
y_pred = []

with torch.no_grad(): # tells do NOT compute gradients
    for images, labels in test_loader: # get batches from the test dataset
        images = images.to(device)
        labels = labels.to(device)
        # if the model is on GPU, the data must also be on GPU
        
        logits = network(images) # The network predicts probabilities for each digit
        preds = torch.argmax(logits, 1) # Convert probabilities to predicted digit

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        # lists now contain all predictions for the test dataset
        '''
        1. Move tensors from GPU -> CPU
        2. Convert tensors -> NumPy arrays
        3. Add them to the lists
        '''

print(classification_report(y_true, y_pred))

# %%
'''
02.26.2026 Test codes
'''
import numpy as np
data = np.arange(3*3).reshape(3,3)
data.shape
np.max(data)
np.max(data, axis=1)
np.max(data, axis=0)
np.sum(data, axis=1)

def f():
    return 1,2

a, b = f()
print(a, b)

_
i
data

outputs
outputs.shape
torch.max(outputs, 1)
torch.argmax(outputs, 1)

len(test_loader) % 64

data_i=2
plt.matshow(images[data_i].numpy().squeeze(0))
outputs[data_i]