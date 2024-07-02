from segmentation_models_pytorch import Unet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch


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

# Load the model
model = Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=21)
model.load_state_dict(torch.load('model/unet-segmentation.pth'))

# Function to visualize predictions
def visualize_predictions(model, dataloader, num_images=5):
    model.eval()
    images, targets = next(iter(dataloader))
    outputs = model(images)
    outputs = torch.argmax(outputs, dim=1)

    fig, axes = plt.subplots(num_images, 3, figsize=(15, num_images * 5))
    for i in range(num_images):
        axes[i, 0].imshow(images[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 0].set_title("Input Image")
        axes[i, 1].imshow(targets[i].squeeze(0).cpu().numpy(), cmap='gray')  # Squeeze the extra dimension
        axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(outputs[i].cpu().numpy(), cmap='gray')
        axes[i, 2].set_title("Predicted Segmentation")

    plt.tight_layout()
    plt.show()

# Visualize some predictions
visualize_predictions(model, dataloader)
