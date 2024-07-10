import torch
import torch.nn as nn
from einops import rearrange

# Definição do Vision Transformer (mesma definição usada no treinamento)
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

# Função para carregar o modelo
def load_model(filepath, device):
    model = VisionTransformer()
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()  # Coloca o modelo em modo de avaliação
    return model

# Exemplo de uso
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('vit_model/vision_transformer.pth', device)

# Agora o modelo está carregado e pronto para fazer previsões
# Exemplo de previsão
dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Substitua por uma imagem real
output = model(dummy_input)
print(output)
