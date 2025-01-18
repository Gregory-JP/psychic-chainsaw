import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms
from einops import repeat
import os
from torch.utils.data import Dataset
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, Tensor
from zeta.nn import SSM
from einops.layers.torch import Reduce



# Pair
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def output_head(dim: int, num_classes: int):
    return nn.Sequential(
        Reduce("b s d -> b d", "mean"),
        nn.LayerNorm(dim),
        nn.Linear(dim, num_classes),
    )

class VisionEncoderMambaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int,
    ):
        super().__init__()
        self.dim = dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        self.forward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.backward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.norm = nn.LayerNorm(dim)
        self.silu = nn.SiLU()
        self.ssm = SSM(dim, dt_rank, dim_inner, d_state)

        self.proj = nn.Linear(dim, dim)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape
        skip = x
        x = self.norm(x)
        z1 = self.proj(x)
        x = self.proj(x)
        x1 = self.process_direction(x, self.forward_conv1d, self.ssm)
        x2 = self.process_direction(x, self.backward_conv1d, self.ssm)
        z = self.silu(z1)
        x1 *= z
        x2 *= z
        return x1 + x2 + skip

    def process_direction(self, x: Tensor, conv1d: nn.Conv1d, ssm: SSM):
        x = rearrange(x, "b s d -> b d s")
        x = self.softplus(conv1d(x))
        x = rearrange(x, "b d s -> b s d")
        x = ssm(x)
        return x

class Vim(nn.Module):
    def __init__(
        self,
        dim: int,
        dt_rank: int = 32,
        dim_inner: int = None,
        d_state: int = None,
        num_classes: int = None,
        image_size: int = 224,
        patch_size: int = 16,
        channels: int = 3,
        dropout: float = 0.1,
        depth: int = 12,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.dropout = dropout
        self.depth = depth

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.to_latent = nn.Identity()
        self.layers = nn.ModuleList([
            VisionEncoderMambaBlock(
                dim=dim,
                dt_rank=dt_rank,
                dim_inner=dim_inner,
                d_state=d_state,
                *args,
                **kwargs,
            ) for _ in range(depth)
        ])
        self.output_head = output_head(dim, num_classes)

    def forward(self, x: Tensor):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.to_latent(x)
        return self.output_head(x)

# Definir o dataset personalizado
class BrainTumorDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform

        # Filtrar apenas as imagens que possuem rótulos
        self.image_files = [
            img for img in sorted(os.listdir(images_dir))
            if os.path.isfile(os.path.join(labels_dir, os.path.splitext(img)[0] + '.txt'))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Obter o caminho da imagem e do rótulo
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, os.path.splitext(self.image_files[idx])[0] + '.txt')

        # Carregar a imagem
        image = Image.open(img_path).convert('RGB')

        # Carregar o rótulo
        with open(label_path, 'r') as f:
            line = f.readline().strip()  # Lê apenas a primeira linha do rótulo
            label = int(line.split()[0])  # Extrai apenas o primeiro valor (classe)

        # Aplicar transformações, se houver
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

if __name__ == '__main__':
    # Configurar device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # Diretórios do dataset
    valid_images_dir = 'data/brain_tumor/valid/images'
    valid_labels_dir = 'data/brain_tumor/valid/labels'

    # Transformações padrão
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Criar dataset e loader de validação
    valid_dataset = BrainTumorDataset(valid_images_dir, valid_labels_dir, transform=transform)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Carregar o modelo
    model = Vim(
        dim=128,
        dt_rank=16,
        dim_inner=128,
        d_state=128,
        num_classes=1,  # Binário: tumor ou não-tumor
        image_size=224,
        patch_size=32,
        channels=3,
        dropout=0.3,
        depth=6
    ).to(device)

    model.load_state_dict(torch.load('models/vim_brain_tumor_model.pth', map_location=device))
    model.eval()
    print("Modelo carregado com sucesso!")

    # Avaliação
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)  # Converte logits para probabilidades
            preds = (probs > 0.1).float()  # Binariza as predições (0 ou 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calcular métricas
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
