import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


# Definição do modelo CustomResNet
class CustomResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomResNet, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# Função para carregar o modelo salvo
def load_model(filepath, device, num_classes=10):
    model = CustomResNet(num_classes=num_classes).to(device)
    state_dict = torch.load(filepath, map_location=device)
    model.load_state_dict(state_dict)
    return model

# Configurações
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print("Usando GPU:", torch.cuda.get_device_name(0))
else:
    print("Usando CPU")

# Carregar o modelo salvo
loaded_model = load_model('custom_resnet.pth', device)
loaded_model.eval()

# Exemplo de previsão com o modelo carregado
dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Substitua por uma imagem real
output = loaded_model(dummy_input)
print(output)
