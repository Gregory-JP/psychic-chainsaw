import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from einops import rearrange

# Definição do Vision Transformer
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224):
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
            nn.Linear(mlp_dim, emb_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=10, emb_size=768, num_heads=12, depth=12, mlp_dim=3072, dropout=0.1):
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
def train(model, loader, criterion, optimizer, device):
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
    # Definição do DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Redimensiona as imagens para 224x224
        transforms.ToTensor(),  # Converte as imagens para tensores
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normaliza as imagens
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Configurações
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print("Usando GPU:", torch.cuda.get_device_name(0))
    else:
        print("Usando CPU")
    
    model = VisionTransformer().to(device)
    
    # Verificar parâmetros do modelo
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    num_epochs = 10

# Loop de Treinamento
    for epoch in range(num_epochs):
        print(f"Iniciando época {epoch+1}/{num_epochs}")
        train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {val_loss}, Accuracy: {val_acc}')

    print("Treinamento concluído.")


"""
Explicação dos Componentes:

    PatchEmbedding: Converte a imagem em patches e aplica uma projeção linear.
    Attention: Implementa o mecanismo de atenção escalonada.
    TransformerBlock: Combina a camada de atenção e uma rede MLP com normalização e dropout.
    VisionTransformer: Junta tudo, empilhando os blocos do transformador e adicionando a camada de classificação final.
"""

# Epoch 10/10, Loss: 1.0975568435443477, Accuracy: 0.5998