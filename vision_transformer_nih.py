import torch
import torchvision.transforms as transforms
from PIL import Image
from vit_model.vit_model_nih_crx8 import VisionTransformer


# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VisionTransformer(num_classes=15).to(device)

# Load the saved weights
model.load_state_dict(torch.load('models/vision_transformer_nih_chest_xray.pth', map_location=device))
model.eval()  # Set the model to evaluation mode

# Define the image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load and preprocess the image
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_transformed = transform(img).unsqueeze(0).to(device)  # Add batch dimension
    return img_transformed

# Example of loading an image for testing
image_path = 'data/nih_cxr8/images/00013118_008.png'
image_tensor = load_image(image_path)

# Run inference
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.sigmoid(output)  # Since it is multi-label classification

# Get the class names
class_names = [
    "No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia"
]

# Display the predicted probabilities
predicted_labels = {class_names[i]: probabilities[0][i].item() for i in range(len(class_names))}

import matplotlib.pyplot as plt

# Print predictions
for label, prob in predicted_labels.items():
    print(f'{label}: {prob:.4f}')

# Visualize the image
img = Image.open(image_path)
plt.imshow(img)
plt.axis('off')
plt.show()
