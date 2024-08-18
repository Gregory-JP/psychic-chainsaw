# Psychic Chainsaw

This repo contains the code and related files to my final paper about image segmentation and MAMBA (SSM).

# Vision Transformer and Mamba UNet for Medical Image Classification

This repository contains implementations of custom deep learning models designed for medical image classification tasks, including the Vision Transformer (ViT) and Mamba UNet architectures. The models are specifically tailored for datasets like NIH Chest X-ray and Chest X-ray Pneumonia, and they leverage state-of-the-art techniques for efficient training and evaluation.

## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Model Architectures](#model-architectures)
- [Training and Evaluation](#training-and-evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project provides implementations of deep learning models tailored for medical image classification. The primary models implemented are:

- **Custom Vision Transformer (ViT):** Adapted for the Chest X-ray Pneumonia dataset.
- **Mamba UNet:** A hybrid model that combines the Vision Transformer with UNet architecture, designed for multi-label classification on the NIH Chest X-ray dataset.

## Datasets

### NIH Chest X-ray Dataset

The NIH Chest X-ray dataset consists of over 100,000 chest X-ray images with labels for 14 common thoracic diseases. The dataset is structured with the following key files:

- **Images:** Stored in the `images` directory.
- **Annotations:** Provided in the `Data_Entry_2017.csv` file.
- **Train/Validation/Test Splits:** Predefined splits are given in the `train_val_list.txt` and `test_list.txt` files.

### Chest X-ray Pneumonia Dataset

The Chest X-ray Pneumonia dataset contains X-ray images classified into two categories: normal and pneumonia. The dataset structure includes:

- **Train, Validation, and Test Folders:** Each folder contains subdirectories for `NORMAL` and `PNEUMONIA` images.

## Model Architectures

### Vision Transformer (ViT)

The Vision Transformer is implemented with the following key features:

- **Patch Embedding:** Converts image patches into embedded vectors.
- **Multi-head Attention:** Captures relationships between different patches.
- **Global Average Pooling:** Reduces the dimensionality for classification tasks.
- **Mixed Precision Training:** Enhances efficiency during training.

### Mamba UNet

The Mamba UNet is a custom architecture that integrates transformer blocks into a UNet-like structure:

- **VMamba Blocks:** Transformer blocks that incorporate multi-head attention and additional convolutional layers.
- **Patch Embedding and Bottleneck Layers:** Capture and process spatial features.
- **Decoder Path:** Mirrors the encoder structure with additional upsampling and convolutional layers for segmentation.

## Training and Evaluation

### Optimizations Implemented

The training process for both models includes several optimizations:

- **Mixed Precision Training:** Reduces memory usage and speeds up computations on GPUs.
- **Gradient Accumulation:** Simulates larger batch sizes to optimize memory usage.
- **Efficient Data Loading:** Utilizes multiple workers and `pin_memory=True` for faster data transfer.
- **Learning Rate Scheduling:** Adaptive learning rate with `StepLR` scheduler.
- **Model Complexity Reduction:** Reduced the depth, embedding size, and number of heads to improve training time.

### Training Configuration

- **Learning Rate:** `3e-4`
- **Batch Size:** `32`
- **Number of Epochs:** `50`
- **Optimizer:** Adam
- **Loss Functions:** 
  - For multi-label classification (NIH dataset): `BCEWithLogitsLoss`
  - For binary classification (Pneumonia dataset): `CrossEntropyLoss`

### Training Procedure

To train the models, simply run:

```bash
python mamba_unet_nih_crx8.py  # For NIH Chest X-ray
python vision_transformer_chest_xray.py  # For Pneumonia dataset
```

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- NVIDIA GPU with CUDA support

### Installing Dependencies

Install the required Python packages using:
`pip install -r requirements.txt`

## Usage

To use the pre-trained models for inference:

```
from mamba_unet_nih_crx8 import load_model  # For Mamba UNet
from vision_transformer_chest_xray import load_model  # For ViT

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('models/mamba_unet_nih_chest_xray_optimized.pth', device)  # Replace with the correct path
model.eval()

# Perform inference
output = model(image_tensor)
```

## Custom Training with Weights and Biases

TO track the training process with Weights and Biases (wandb):
1. Set up your 'wandb' account and API key.
2. Modify the script to include your 'wandb' credentials.
3. Run the training script to automatically log metrics to 'wandb'.

## Results
The results from the training and evaluation will be plotted and saved as figures. These include:
- Loss over Epochs: Plot showing the training and validation loss.
- Accuracy over Epochs: Plots showing the validation accuracy.

## Licença

This project is licensed under the MIT License -see the LICENSE file for details.