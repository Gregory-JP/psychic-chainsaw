import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from einops import rearrange
import torch.cuda.amp as amp
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import wandb
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


from metrics.metric_tracker_class import MetricTracker


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

# Definição do Vision Transformer
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=512, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_size, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.pos_emb = nn.Parameter(torch.zeros((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)  # (B, emb_size, H/P, W/P)
        x = rearrange(x, 'b e (h) (w) -> b (h w) e')  # (B, N, emb_size)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, emb_size)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, emb_size)
        x += self.pos_emb
        return x

class Attention(nn.Module):
    def __init__(self, emb_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.fc_out = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.attention = Attention(emb_size, num_heads)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),  # Adicionando dropout aqui
            nn.Linear(mlp_dim, emb_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=10, emb_size=512, num_heads=8, depth=6, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.transformer = nn.Sequential(
            *[TransformerBlock(emb_size, num_heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        x = self.head(x)
        return x

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
    from load_env import wandb_key

    api_key = wandb_key()

    # Configurações
    config = {
        'learning_rate': 3e-4,
        'batch_size': 32,
        'num_epochs': 50,
        'optimizer': 'Adam',
        'loss_function': 'CrossEntropyLoss',
        'model': 'Vision Transformer',
        'dataset': 'CIFAR-10'
    }

    # Autenticação no wandb
    wandb.login(key=api_key)

    # Definição do DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Redimensiona as imagens para 224x224
        transforms.RandomHorizontalFlip(),  # Aplica flip horizontal aleatório
        transforms.RandomRotation(10),  # Rotação aleatória de até 10 graus
        transforms.ToTensor(),  # Converte as imagens para tensores
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normaliza as imagens
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Configurações
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print("Usando GPU:", torch.cuda.get_device_name(0))
    else:
        print("Usando CPU")
    
    model = VisionTransformer().to(device)
    scaler = amp.GradScaler()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    num_epochs = 50

    # Armazenar métricas
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Inicializar o wandb
    setup_wandb(project_name='Cifar-10', run_name='Vision Transformer', config=config)

    # Instancia o rastreador de métricas e configura o dispositivo
    metric_tracker = MetricTracker(device, task='multiclass', num_classes=10)

    # Loop de Treinamento
    for epoch in range(num_epochs):
        print(f"Iniciando época {epoch+1}/{num_epochs}")
        
        # Resetar as métricas no início da época
        metric_tracker.reset()

        # Treinamento
        train_loss = 0
        model.train()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            # Atualizar a métrica de acurácia do treinamento
            metric_tracker.update_train(outputs, labels)

        # Avaliação
        val_loss = 0
        model.eval()

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Atualizar a métrica de acurácia da validação
                metric_tracker.update_val(outputs, labels)

        # Computar acurácia final
        train_acc = metric_tracker.compute_train()
        val_acc = metric_tracker.compute_val()

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(test_loader))
        val_accuracies.append(val_acc.item())  # Armazenar o valor da acurácia

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss / len(train_loader)}, Train Accuracy: {train_acc}, Val Loss: {val_loss / len(test_loader)}, Val Accuracy: {val_acc}')

        scheduler.step()

        # Logging no wandb
        log_metrics(epoch, train_loss / len(train_loader), val_loss / len(test_loader), val_acc)

    # Conclusão do treinamento
    print("Treinamento concluído.")

    # Salvando o modelo treinado
    torch.save(model.state_dict(), 'models/vision_transformer_cifar10.pth')
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
