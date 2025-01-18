import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from vision_mamba.vision_mamba_nih_crx8 import Vim


# Instanciar o modelo
model = Vim(
    dim=128,
    dt_rank=16,
    dim_inner=128,
    d_state=128,
    num_classes=15,  # 15 classes do NIH Chest X-ray
    image_size=224,
    patch_size=32,
    channels=3,
    dropout=0.1,
    depth=6
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Carregar os pesos salvos
model.load_state_dict(torch.load('models/vim_nih_chest_xray_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Transformações da imagem de entrada
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Mapeamento de classes
class_names = [
    "No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia"
]

# Função para fazer inferência em uma nova imagem
def predict_image(image_path, model, transform):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)  # Adicionar o batch dimension

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.sigmoid(output).squeeze(0)  # Aplicar sigmoid para multi-label
    return probabilities

# Carregar e prever uma imagem
image_path = 'data/nih_cxr8/images/00030791_000.png'
probabilities = predict_image(image_path, model, transform)

# Exibir as probabilidades previstas para cada classe
predicted_labels = {class_names[i]: probabilities[i].item() for i in range(len(class_names))}

# Imprimir as predições
for label, prob in predicted_labels.items():
    print(f'{label}: {prob:.4f}')

# Exibir a imagem e as predições mais prováveis
img = Image.open(image_path)

plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.title("NIH Chest X-ray")
plt.axis('off')
plt.show()
