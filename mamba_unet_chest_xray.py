import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from mamba_unet.mamba_unet_chest_xray import MambaSSMUNet  # Importando o modelo MambaSSMUNet

# Carregar o modelo MambaSSMUNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MambaSSMUNet(num_classes=2).to(device)

# Carregar os pesos salvos
model.load_state_dict(torch.load('models/mamba_unet_chest_xray.pth', map_location=device))
model.eval()  # Definir o modelo em modo de avaliação

# Definir as transformações de imagem (iguais às usadas no treinamento)
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
        probabilities = torch.softmax(output, dim=1)  # Softmax para obter as probabilidades de classificação
    return probabilities

# Caminho da imagem de teste (substituir pelo caminho da imagem real)
image_path = r'data\chest_xray\val\PNEUMONIA\person1946_bacteria_4875.jpeg'

# Carregar e pré-processar a imagem
test_image = load_image(image_path)

# Realizar a predição
output_probs = predict(model, test_image)

# Mapeamento de índices de classes para rótulos
class_names = ["Normal", "Pneumonia"]

# Exibir as probabilidades previstas para cada classe
predicted_labels = {class_names[i]: output_probs[0][i].item() for i in range(len(class_names))}

# Imprimir as predições
for label, prob in predicted_labels.items():
    print(f'{label}: {prob:.4f}')

# Visualizar a imagem e as predições
img = Image.open(image_path)
plt.imshow(img)
plt.axis('off')
plt.title(f"Predição: {class_names[torch.argmax(output_probs, dim=1).item()]}")
plt.show()
