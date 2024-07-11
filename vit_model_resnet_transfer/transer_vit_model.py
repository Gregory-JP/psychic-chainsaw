import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
import torch.cuda.amp as amp
import matplotlib.pyplot as plt


# Definição do modelo CustomResNet
class CustomResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomResNet, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# Função para carregar o modelo salvo
def load_model(filepath, device, num_classes=2):
    model = CustomResNet(num_classes=num_classes).to(device)
    state_dict = torch.load(filepath, map_location=device)
    model.load_state_dict(state_dict)
    return model

# Função de Treinamento
def train(model, loader, criterion, optimizer, device, scaler):
    model.train()
    train_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        with amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
    return train_loss / len(loader)

# Função de Avaliação
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


if __name__ == '__main__':

    # Configurações
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        print("Usando GPU:", torch.cuda.get_device_name(0))
    else:
        print("Usando CPU")

    # Transformações para os dados de treinamento e validação
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Carregar os datasets
    train_dataset = datasets.ImageFolder(root='data/chest_xray/train', transform=transform_train)
    val_dataset = datasets.ImageFolder(root='data/chest_xray/val', transform=transform_val)
    test_dataset = datasets.ImageFolder(root='data/chest_xray/test', transform=transform_val)

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Instanciar o modelo
    model = CustomResNet(num_classes=2).to(device)
    scaler = amp.GradScaler()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    num_epochs = 50

    # Armazenar métricas
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Loop de Treinamento
    for epoch in range(num_epochs):
        print(f"Iniciando época {epoch+1}/{num_epochs}")
        train_loss = train(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, Accuracy: {val_acc}')
        scheduler.step()


    print("Treinamento concluído.")

    torch.save(model.state_dict(), 'vit_model_resnet_transfer/custom_resnet_pneumonia.pth')
    print("Modelo salvo com sucesso.")

    # Plotar as métricas
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs (chest_xray)')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs (chest_xray)')

    plt.tight_layout()
    plt.show()

    # # Carregar o modelo salvo e testar
    # loaded_model = load_model('vit_model_resnet_transfer/custom_resnet_pneumonia.pth', device)
    # loaded_model.eval()

    # # Exemplo de previsão com o modelo carregado
    # dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Substitua por uma imagem real
    # output = loaded_model(dummy_input)
    # print(output)
