from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights
from torchvision import transforms, models
import matplotlib.pyplot as plt
import torch.cuda.amp as amp
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import pandas as pd
import torch
import os

import wandb

# Funções para configurar e usar o wandb
def setup_wandb(project_name, run_name, config):
    wandb.init(project=project_name, name=run_name, config=config)

def log_metrics(epoch, train_loss, val_loss, val_accuracy):
    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy
    })


# Definição do modelo CustomResNet
class CustomResNet(nn.Module):
    def __init__(self, num_classes=15):
        super(CustomResNet, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# Dataset personalizado para NIH Chest X-rays
class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, file_list=None):
        self.annotations = pd.read_csv(csv_file)
        if file_list:
            with open(file_list, 'r') as f:
                files = f.read().splitlines()
            self.annotations = self.annotations[self.annotations['Image Index'].isin(files)]
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = [
            "No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
            "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
            "Fibrosis", "Pleural_Thickening", "Hernia"
        ]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        labels = self.annotations.iloc[idx, 1].split('|')
        label_tensor = torch.zeros(len(self.class_names))

        for label in labels:
            if label in self.class_names:
                label_tensor[self.class_names.index(label)] = 1
        if self.transform:
            image = self.transform(image)
        
        return image, label_tensor

# Função para carregar o modelo salvo
def load_model(filepath, device, num_classes=15):
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
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
    
    return total_loss / len(loader), correct / (len(loader.dataset) * len(loader.dataset[0][1]))


if __name__ == '__main__':
    # Configurações
    config = {
        'learning_rate': 3e-4,
        'batch_size': 32,
        'num_epochs': 20,
        'optimizer': 'Adam',
        'loss_function': 'BCEWithLogitsLoss',
        'model': 'ResNet18',
        'dataset': 'NIH Chest X-rays'
    }

    # Autenticação no wandb
    wandb.login(key='79d389395f5ad034f60cd189e8d6b583b5061b5a')

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
    train_dataset = NIHChestXrayDataset(
        csv_file='data/nih_cxr8/Data_Entry_2017.csv',
        root_dir=r'data\nih_cxr8\images',
        transform=transform_train,
        file_list=r'data\nih_cxr8\train_val_list.txt'
    )
    val_dataset = NIHChestXrayDataset(
        csv_file='data/nih_cxr8/Data_Entry_2017.csv',
        root_dir=r'data\nih_cxr8\images',
        transform=transform_val,
        file_list=r'data\nih_cxr8\train_val_list.txt'
    )
    test_dataset = NIHChestXrayDataset(
        csv_file='data/nih_cxr8/Data_Entry_2017.csv',
        root_dir=r'data\nih_cxr8\images',
        transform=transform_val,
        file_list=r'data\nih_cxr8\test_list.txt'
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Instanciar o modelo
    model = CustomResNet(num_classes=15).to(device)
    scaler = amp.GradScaler()

    criterion = nn.BCEWithLogitsLoss()  # Loss para classificação multi-label
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    num_epochs = 50

    # Armazenar métricas
    train_losses = []
    val_losses = []
    val_accuracies = []

    setup_wandb(project_name='NIH-Chest-Xrays', run_name='ResNet18_Experiment', config=config)

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

        log_metrics(epoch, train_loss, val_loss, val_acc)

    print("Treinamento concluído.")
    
    torch.save(model.state_dict(), 'models/custom_resnet_nih_chest_xray.pth')
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
    plt.title('Loss over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')

    plt.tight_layout()
    plt.show()
    plt.savefig('results/vit_RT_result_nih_chest_xray.png')

    # # Carregar o modelo salvo e testar
    # loaded_model = load_model('custom_resnet_nih_chest_xray.pth', device)
    # loaded_model.eval()

    # # Exemplo de previsão com o modelo carregado
    # dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Substitua por uma imagem real
    # output = loaded_model(dummy_input)
    # print(output)
