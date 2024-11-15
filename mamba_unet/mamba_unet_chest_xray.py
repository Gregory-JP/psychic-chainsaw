import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from einops import rearrange
import torch.cuda.amp as amp
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from PIL import Image
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

# Definição do MambaUNet
class StateSpaceBlock(nn.Module):
    def __init__(self, emb_size=256, num_heads=4, mlp_dim=512, dropout=0.1):
        super(StateSpaceBlock, self).__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.attention = nn.MultiheadAttention(embed_dim=emb_size, num_heads=num_heads, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_size),
            nn.Dropout(dropout),
        )
        self.state_update = nn.Linear(emb_size, emb_size)
        self.conv = nn.Conv1d(emb_size, emb_size, kernel_size=3, padding=1, groups=emb_size)
        self.silu = nn.SiLU()

    def forward(self, x, state):
        state = self.state_update(state) + x
        normed_x = self.norm1(x)
        normed_x_t = normed_x.transpose(0, 1)
        attn_output, _ = self.attention(normed_x_t, normed_x_t, normed_x_t)
        attn_output = attn_output.transpose(0, 1)
        x = x + attn_output
        res = x
        x = x.transpose(1, 2)
        x = self.silu(self.conv(x))
        x = x.transpose(1, 2)
        x = res + x + state
        x = x + self.mlp(self.norm2(x))
        return x, state

class MambaSSMUNet(nn.Module):
    def __init__(self, img_size=112, patch_size=16, in_channels=3, num_classes=2, emb_size=256, num_heads=4, depth=4, mlp_dim=512, dropout=0.1):
        super(MambaSSMUNet, self).__init__()
        self.patch_embed = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2, emb_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.initial_state = nn.Parameter(torch.zeros(1, emb_size))
        self.encoder = nn.ModuleList(
            [StateSpaceBlock(emb_size, num_heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.bottleneck = StateSpaceBlock(emb_size, num_heads, mlp_dim, dropout)
        self.decoder = nn.ModuleList(
            [StateSpaceBlock(emb_size, num_heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.head = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        x = rearrange(x, 'b e h w -> b (h w) e')
        x += self.pos_emb
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        state = self.initial_state.expand(B, x.size(1), -1).clone()
        for layer in self.encoder:
            x, state = layer(x, state)
        x, state = self.bottleneck(x, state)
        for layer in self.decoder:
            x, state = layer(x, state)
        x = self.head(x[:, 0])
        return x

# Dataset personalizado
class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.images = []
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for img in os.listdir(cls_path):
                self.images.append((os.path.join(cls_path, img), cls))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, cls = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[cls]
        if self.transform:
            image = self.transform(image)
        return image, label

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
    # from load_env import wandb_key

    # key = wandb_key()

    # Configurações
    config = {
        'learning_rate': 3e-4,
        'batch_size': 32,
        'num_epochs': 50,
        'optimizer': 'Adam',
        'loss_function': 'CrossEntropyLoss',
        'model': 'Mamba SSM UNet',
        'dataset': 'Chest X-ray Pneumonia'
    }

    wandb.login(key='')

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = ChestXrayDataset(root_dir='data/chest_xray/train', transform=transform)
    val_dataset = ChestXrayDataset(root_dir='data/chest_xray/val', transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MambaSSMUNet(img_size=112, num_classes=2).to(device)
    scaler = amp.GradScaler()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    setup_wandb(project_name='Chest-Xray-Pneumonia', run_name='MambaUNet', config=config)

    for epoch in range(config['num_epochs']):
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        train_loss = train(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        log_metrics(epoch, train_loss, val_loss, val_acc)
        scheduler.step()

    torch.save(model.state_dict(), 'mamba_unet_chest_xray.pth')
    print("Modelo salvo com sucesso!")
