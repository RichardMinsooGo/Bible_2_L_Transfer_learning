'''
Data Engineering
'''

'''
D1. Import Libraries for Data Engineering
'''
import numpy as np

import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

'''
D2. Load CIFAR10 data
'''
train_dataset = datasets.CIFAR10(root='./data', 
                             download=True,
                             train=True,
                             transform=transforms.ToTensor())

test_dataset = datasets.CIFAR10(root='./data', 
                            download=True,
                            train=False,
                             transform=transforms.ToTensor())

'''
Model Engineering
'''

'''
M1. Import Libraries for Model Engineering
'''

import torch.nn as nn
import torch.optim as optimizers

np.random.seed(123)
torch.manual_seed(123)

'''
M2. Set Hyperparameters
'''
hidden_size = 256 
output_dim = 10 # output layer dimensionality = num_classes
EPOCHS = 5
batch_size = 100
learning_rate = 0.001

'''
M3. DataLoader
'''

train_ds = torch.utils.data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size, 
                                       shuffle=True)
test_ds = torch.utils.data.DataLoader(dataset=test_dataset,
                                      batch_size=batch_size, 
                                      shuffle=False)

# Specify the image classes
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
          "horse", "ship", "truck"]

'''
M4. Build NN model
'''
import torchvision.models as models

    
"""    
# Alexnet Input size is too small, try it at custom dataset
# transfer_model = models.alexnet(pretrained=True, progress=True)

transfer_model = models.vgg16(pretrained=True, progress=True)
transfer_model = models.vgg19(pretrained=True, progress=True)


transfer_model = models.googlenet(pretrained=True)

# Inception Input size is too small, try it at custom dataset
transfer_model = models.inception_v3(pretrained=True)

transfer_model = models.mobilenet_v2(pretrained=True)

transfer_model = models.resnet18(pretrained=False, progress=True)
transfer_model = models.resnet34(pretrained=True, progress=True)
transfer_model = models.resnet50(pretrained=True, progress=True)
transfer_model = models.resnet101(pretrained=True, progress=True)
transfer_model = models.resnet152(pretrained=False, progress=True)

"""
'''
M5. Transfer model to GPU
'''

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transfer_model =models.resnet152(pretrained=True, progress=True)
model=transfer_model.to(device)

'''
M6. Optimizer
'''
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

'''
M7. Define Loss Function
'''

criterion = nn.CrossEntropyLoss()
'''
M8. Define train loop
'''

def train_step(model, images, labels):
    model.train()
    
    images = images.to(device)
    labels = labels.to(device)

    # Forward pass
    predictions = model(images)
    loss = criterion(predictions, labels)
    loss_val = loss.item()

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Pytorch need a manual coding for accuracy
    # max returns (value ,index)
    _, predicted = torch.max(predictions.data, 1)
    n_samples = labels.size(0)
    n_correct = (predicted == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    
    return loss_val, acc

'''
M9. Define validation / test loop
'''

def test_step(model, images, labels):
    model.eval()
    
    images = images.to(device)
    labels = labels.to(device)

    # Forward pass
    predictions = model(images)
    loss = criterion(predictions, labels)
    loss_val = loss.item()

    # Pytorch need a manual coding for accuracy
    # max returns (value ,index)
    _, predicted = torch.max(predictions.data, 1)
    n_samples = labels.size(0)
    n_correct = (predicted == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    
    return loss_val, acc

'''
M10. Define Episode / each step process
'''
from tqdm import tqdm, tqdm_notebook, trange

for epoch in range(EPOCHS):
    
    with tqdm_notebook(total=len(train_ds), desc=f"Train Epoch {epoch+1}") as pbar:    
        train_losses = []
        train_accuracies = []
        
        for i, (images, labels) in enumerate(train_ds):
         
            loss_val, acc = train_step(model, images, labels)
            
            train_losses.append(loss_val)
            train_accuracies.append(acc)
            
            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.4f} ({np.mean(train_losses):.4f}) Acc: {acc:.3f} ({np.mean(train_accuracies):.3f})")


'''
M11. Model evaluation
'''
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():

    with tqdm_notebook(total=len(test_ds), desc=f"Test_ Epoch {epoch+1}") as pbar:    
        test_losses = []
        test_accuracies = []

        for images, labels in test_ds:
            loss_val, acc = test_step(model, images, labels)

            test_losses.append(loss_val)
            test_accuracies.append(acc)

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.4f} ({np.mean(test_losses):.4f}) Acc: {acc:.3f} ({np.mean(test_accuracies):.3f})")
            
