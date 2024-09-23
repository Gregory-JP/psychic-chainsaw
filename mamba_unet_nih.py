import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from mamba_unet.mamba_unet_nih_crx8 import MambaSSMUNet

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MambaSSMUNet(num_classes=15).to(device)

# Load the saved weights
model.load_state_dict(torch.load('models/mamba_unet_nih_chest_xray.pth', map_location=device))
model.eval()

# Define image transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load the test image
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_transformed = transform(img).unsqueeze(0).to(device)  # Add batch dimension
    return img_transformed

# Perform inference
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.sigmoid(output)  # Apply sigmoid for multi-label classification
    return probabilities

# Test image path (replace with the actual image path)
image_path = 'data/nih_cxr8/images/00030791_000.png'

# Load and preprocess the image
test_image = load_image(image_path)

# Make prediction
output_probs = predict(model, test_image)

# Map class indices to labels
class_names = [
    "No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia"
]

# Display the predicted probabilities
predicted_labels = {class_names[i]: output_probs[0][i].item() for i in range(len(class_names))}

# Visualize the image and predictions
img = Image.open(image_path)
plt.imshow(img)
plt.axis('off')
plt.show()

# Print the predictions
for label, prob in predicted_labels.items():
    print(f'{label}: {prob:.4f}')
