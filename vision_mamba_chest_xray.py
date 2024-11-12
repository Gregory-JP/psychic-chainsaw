import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from vision_mamba.vision_mamba_chest_xray import Vim  # Importando o modelo Vim

# Instanciar o modelo VIM
model = Vim(
    dim=128,
    dt_rank=16,
    dim_inner=128,
    d_state=128,
    num_classes=2,  # Duas classes: 'Normal' e 'Pneumonia'
    image_size=224,
    patch_size=32,
    channels=3,
    dropout=0.1,
    depth=6
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Carregar os pesos salvos
model.load_state_dict(torch.load('models/vim_chest_xray_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Transformações da imagem de entrada
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Função para fazer inferência em uma nova imagem
def predict_image(image_path, model, transform):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)  # Adicionar o batch dimension
    with torch.no_grad():
        output = model(image_tensor)
    _, predicted = torch.max(output, 1)
    return predicted.item()

# Carregar e prever uma imagem
image_path = 'data/chest_xray/val/PNEUMONIA/person1946_bacteria_4874.jpeg'
prediction = predict_image(image_path, model, transform)

# Exibir a imagem e o resultado da predição
class_names = ['Normal', 'Pneumonia']  # As classes do dataset
pred_class = class_names[prediction]

# Exibir a imagem e o resultado
plt.figure(f'{image_path}', figsize=(6, 6))
plt.imshow(Image.open(image_path))
plt.title(f'Predicted: {pred_class}')
plt.axis('off')  # Desativar os eixos
plt.show()

# Caminho para o diretório de imagens
image_directory = 'data/chest_xray/val/NORMAL/'

# Carregar todas as imagens do diretório para exibir em um grid
images = []
predictions = []

for filename in os.listdir(image_directory):
    if filename.endswith(".jpeg"):
        img_path = os.path.join(image_directory, filename)
        prediction = predict_image(img_path, model, transform)
        pred_class = class_names[prediction]
        print(f'Predicted: {pred_class} for {filename}')
        
        img = Image.open(img_path)
        images.append(img)
        predictions.append(pred_class)  # Armazenar a predição para cada imagem

# Configurar o grid para visualização
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))

# Exibir as imagens e previsões no grid
for ax, img, pred in zip(axes.flat, images, predictions):
    ax.imshow(img)
    ax.axis('off')
    # Adicionar o texto da predição sobre a imagem
    ax.text(10, 20, f'Predicted: {pred}', color='white', fontsize=12, 
            bbox=dict(facecolor='black', alpha=0.5))

plt.tight_layout()
plt.show()
