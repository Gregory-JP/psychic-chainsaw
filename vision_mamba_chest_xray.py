from vision_mamba.vision_mamba_chest_xray import Vim
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import torch


# Definir as transformações de pré-processamento
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Carregar o modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Vim(
    dim=128,  
    dt_rank=16,
    dim_inner=128,
    d_state=128,
    num_classes=2,  
    image_size=224,
    patch_size=32,
    channels=3,
    dropout=0.1,
    depth=6  
).to(device)

model.load_state_dict(torch.load('models/vim_chest_xray_model.pth', map_location=device))
model.eval()


# Função para predição
def predict_and_show_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_image)
        _, predicted = torch.max(output, 1)
        label = "Pneumonia" if predicted.item() == 1 else "Normal"

    # Exibir a imagem
    plt.imshow(image)
    plt.axis('off')  # Esconder os eixos
    plt.title(f"Predição: {label}", fontsize=16, color='blue')  # Rótulo previsto
    plt.show()
    print(f"Predição: {label}")


# Exemplo de uso
image_path = r'data\chest_xray\val\PNEUMONIA\person1954_bacteria_4886.jpeg'
predict_and_show_image(image_path)
