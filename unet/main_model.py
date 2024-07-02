import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.losses import DiceLoss
import numpy as np
import matplotlib.pyplot as plt

# Load the VOCSegmentation dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

target_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.VOCSegmentation(
    root="./data",
    year="2012",
    image_set="train",
    download=True,
    transform=transform,
    target_transform=target_transform,
)

# Prepare the dataset
def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    targets = torch.stack(targets).long()  # Convert targets to long type for loss calculation
    return images, targets

dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)

# Define the U-Net model
model = Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=21)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = DiceLoss(mode='multiclass')

# Define the training loop
def train(model, dataloader, optimizer, loss_fn):
    model.train()
    for images, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f'Loss: {loss.item()}')


# Train the model
train(model, dataloader, optimizer, loss_fn)

# Save the model
torch.save(model.state_dict(), 'model/unet-segmentation.pth')
