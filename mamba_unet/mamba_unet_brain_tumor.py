import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from einops import rearrange
import torch.cuda.amp as amp
from PIL import Image
import os


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
class BrainTumorDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = [
            img for img in sorted(os.listdir(images_dir))
            if os.path.isfile(os.path.join(labels_dir, os.path.splitext(img)[0] + '.txt'))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, os.path.splitext(self.image_files[idx])[0] + '.txt')

        # Carregar a imagem
        image = Image.open(img_path).convert('RGB')

        # Carregar o rótulo
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            label = int(line.split()[0])  # Extrai apenas a classe (0 ou 1)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

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
    # Transformações
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Diretórios do dataset
    train_images_dir = 'data/brain_tumor/train/images'
    train_labels_dir = 'data/brain_tumor/train/labels'
    valid_images_dir = 'data/brain_tumor/valid/images'
    valid_labels_dir = 'data/brain_tumor/valid/labels'

    # Criar datasets e dataloaders
    train_dataset = BrainTumorDataset(train_images_dir, train_labels_dir, transform=transform)
    valid_dataset = BrainTumorDataset(valid_images_dir, valid_labels_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Configurar o dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # Instanciar o modelo
    model = MambaSSMUNet(img_size=112, num_classes=2).to(device)
    scaler = amp.GradScaler()

    # Função de perda e otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # Treinamento
    num_epochs = 50
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = evaluate(model, valid_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    # Salvar o modelo
    torch.save(model.state_dict(), 'models/mamba_unet_brain_tumor.pth')
    print("Modelo salvo com sucesso!")
