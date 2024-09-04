import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
from vit_model.vit_model_chest_xray import VisionTransformer

# Instanciar o modelo
model = VisionTransformer(num_classes=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Carregar os pesos salvos
model.load_state_dict(torch.load('models/vision_transformer_chest_xray.pth'))
model = model.to(device)
model.eval()

# Transformações da imagem de entrada
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
image_path = 'data/chest_xray/val/PNEUMONIA/person1946_bacteria_4874.jpeg'  # Insira o caminho da imagem aqui
prediction = predict_image(image_path, model, transform)

# Exibir a imagem e o resultado da predição
class_names = ['Normal', 'Pneumonia']  # As classes do dataset
pred_class = class_names[prediction]

# Exibir a imagem e o resultado
plt.imshow(Image.open(image_path))
plt.title(f'Predicted: {pred_class}')
plt.show()

# # Path to your image directory
# image_directory = ''

# # Load all images from the directory for a grid
# images = []
# for filename in os.listdir(image_directory):
#     if filename.endswith(".jpeg"):
#         img_path = os.path.join(image_directory, filename)
#         prediction = predict_image(img_path, model, transform)
#         pred_class = class_names[prediction]
#         print(f'Predicted: {pred_class} for {filename}')
#         img = Image.open(img_path)
#         images.append(img)

# # Configure the grid for visualization
# fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))

# # Display the images in the grid
# for ax, img in zip(axes.flat, images):
#     ax.imshow(img)
#     ax.axis('off')

# plt.tight_layout()
# plt.show()
