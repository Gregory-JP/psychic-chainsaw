import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from mamba_unet.mamba_unet_nih_crx8 import MambaSSMUNet  # Importando o modelo MambaSSMUNet

# Carregar o modelo MambaSSMUNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MambaSSMUNet(num_classes=15).to(device)

# Carregar os pesos salvos
model.load_state_dict(torch.load('models/mamba_unet_nih_chest_xray.pth', map_location=device, weights_only=True))
model.eval()  # Definir o modelo em modo de avaliação

# Definir as transformações de imagem (iguais às usadas no treinamento)
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Função para carregar e pré-processar a imagem
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_transformed = transform(img).unsqueeze(0).to(device)  # Adicionar dimensão do batch
    return img_transformed

# Função para realizar inferência
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.sigmoid(output)  # Aplicar sigmoid para classificação multi-label
    return probabilities

# Caminho da imagem de teste (substituir pelo caminho da imagem real)
image_path = 'data/nih_cxr8/images/00030791_000.png'

# Carregar e pré-processar a imagem
test_image = load_image(image_path)

# Realizar a predição
output_probs = predict(model, test_image)

# Mapeamento de índices de classes para rótulos
class_names = [
    "No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia"
]

# Exibir as probabilidades previstas para cada classe
predicted_labels = {class_names[i]: output_probs[0][i].item() for i in range(len(class_names))}

# Imprimir as predições
for label, prob in predicted_labels.items():
    print(f'{label}: {prob:.4f}')

# Visualizar a imagem e as predições
img = Image.open(image_path)
plt.imshow(img)
plt.axis('off')
plt.show()
