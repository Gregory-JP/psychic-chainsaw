import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch.cuda.amp as amp
from torch.optim.lr_scheduler import StepLR
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

class StateSpaceBlock(nn.Module):
    def __init__(self, emb_size=256, num_heads=4, mlp_dim=512, dropout=0.1):
        super(StateSpaceBlock, self).__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        
        # Mecanismo de atenção multi-cabeça (mantido do modelo original)
        self.attention = nn.MultiheadAttention(embed_dim=emb_size, num_heads=num_heads, dropout=dropout)

        # MLP para processamento não-linear
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_size),
            nn.Dropout(dropout),
        )
        
        # Inicialização do estado latente (uma camada Linear simples para atualização)
        self.state_update = nn.Linear(emb_size, emb_size)
        self.conv = nn.Conv2d(emb_size, emb_size, kernel_size=3, padding=1, groups=emb_size)
        self.silu = nn.SiLU()

    def forward(self, x, state):
        # Atualizamos o estado latente com base na entrada atual
        state = self.state_update(state) + x

        # Normalização e atenção
        normed_x = self.norm1(x)
        attn_output, _ = self.attention(normed_x, normed_x, normed_x)
        x = x + attn_output
        
        # Reorganização e convolução
        res = x
        x = rearrange(x, 'b n e -> b e n 1')
        x = self.silu(self.conv(x))
        x = rearrange(x, 'b e n 1 -> b n e')
        
        # Atualização final usando a MLP e o estado latente atualizado
        x = res + x + state
        x = x + self.mlp(self.norm2(x))
        return x, state

class MambaSSMUNet(nn.Module):
    def __init__(self, img_size=112, patch_size=16, in_channels=3, num_classes=15, emb_size=256, num_heads=4, depth=4, mlp_dim=512, dropout=0.1):
        super(MambaSSMUNet, self).__init__()

        # Patch embedding (dividindo imagem em pequenos patches)
        self.patch_embed = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2, emb_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))

        # Inicializando o estado latente (zerado inicialmente)
        self.initial_state = nn.Parameter(torch.zeros(1, emb_size))

        # Encoder usando blocos baseados em SSM
        self.encoder = nn.ModuleList(
            [StateSpaceBlock(emb_size, num_heads, mlp_dim, dropout) for _ in range(depth)]
        )

        # Bottleneck (um bloco SSM no meio da arquitetura)
        self.bottleneck = StateSpaceBlock(emb_size, num_heads, mlp_dim, dropout)

        # Decoder (reconstruindo a imagem)
        self.decoder = nn.ModuleList(
            [StateSpaceBlock(emb_size, num_heads, mlp_dim, dropout) for _ in range(depth)]
        )

        # Cabeçalho para gerar a saída final (com o número de classes)
        self.head = nn.Conv2d(emb_size, num_classes, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Embed de patches
        x = self.patch_embed(x)
        x = rearrange(x, 'b e h w -> b (h w) e')
        x += self.pos_emb

        # Incluímos o token de classe (CLS token)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)

        # Inicializamos o estado latente
        state = self.initial_state.expand(B, -1)

        # Passamos o estado latente através do encoder
        for layer in self.encoder:
            x, state = layer(x, state)

        # Aplicamos o bottleneck (central)
        x, state = self.bottleneck(x, state)

        # Passamos o estado pelo decoder
        for layer in self.decoder:
            x, state = layer(x, state)

        # Reconstrução da imagem
        x = rearrange(x[:, 1:], 'b (h w) e -> b e h w', h=H // 16, w=W // 16)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        x = self.head(x)

        # Pooling global
        x = torch.mean(x, dim=[2, 3])  # Redução de H e W para 1x1
        return x

# Dataset and Dataloaders
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

def train(model, loader, criterion, optimizer, device, scaler, accumulation_steps=4):
    model.train()
    train_loss = 0
    optimizer.zero_grad()
    
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        with amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        train_loss += loss.item() * accumulation_steps
    return train_loss / len(loader)

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
    from load_env import wandb_key

    api_key = wandb_key()

    config = {
        'learning_rate': 3e-4,
        'batch_size': 32,
        'num_epochs': 50,
        'optimizer': 'Adam',
        'loss_function': 'BCEWithLogitsLoss',
        'model': 'Mamba UNet',
        'dataset': 'NIH Chest X-rays'
    }

    wandb.login(key=api_key)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MambaSSMUNet(num_classes=15).to(device)
    scaler = amp.GradScaler()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    # Updated transform to reduce image size to 112x112
    transform_train = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = NIHChestXrayDataset(
        csv_file='data/nih_cxr8/Data_Entry_2017.csv',
        root_dir=r'data\nih_cxr8\images',
        transform=transform_train,
        file_list=r'data\nih_cxr8\train_val_list.txt'
    )

    val_dataset = NIHChestXrayDataset(
    csv_file='data/nih_cxr8/Data_Entry_2017.csv',
    root_dir=r'data\nih_cxr8\images',
    transform=transform_train,
    file_list=r'data\nih_cxr8\train_val_list.txt'
    )
    test_dataset = NIHChestXrayDataset(
        csv_file='data/nih_cxr8/Data_Entry_2017.csv',
        root_dir=r'data\nih_cxr8\images',
        transform=transform_train,
        file_list=r'data\nih_cxr8\test_list.txt'
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    num_epochs = 50
    train_losses = []
    val_losses = []
    val_accuracies = []

    setup_wandb(project_name='NIH-Chest-Xrays', run_name='Mamba_Unet_Optimized', config=config)

    for epoch in range(num_epochs):
        print(f"Iniciando época {epoch+1}/{num_epochs}")
        train_loss = train(model, train_loader, criterion, optimizer, device, scaler, accumulation_steps=4)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, Accuracy: {val_acc}')
        scheduler.step()

        log_metrics(epoch, train_loss, val_loss, val_acc)

    torch.save(model.state_dict(), 'models/mamba_unet_nih_chest_xray.pth')
    print("Modelo salvo com sucesso.")

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