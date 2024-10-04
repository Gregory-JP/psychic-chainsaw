import torch
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

if __name__ == '__main__':
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import torch.nn as nn

    # Transformações para os dados
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Redimensionar para o tamanho necessário
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Carregar o conjunto de dados CIFAR-10
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Instanciar o modelo
    model = Vim(
        dim=128,  # Diminuir a dimensão para economizar memória
        dt_rank=16,
        dim_inner=128,
        d_state=128,
        num_classes=10,
        image_size=224,
        patch_size=32,  # Certificar-se de que 224 é divisível por 32
        channels=3,
        dropout=0.1,
        depth=6  # Reduzir a profundidade para diminuir o consumo de memória
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5  # Comece com menos épocas para testes
    accumulation_steps = 1  # Não acumular gradientes inicialmente

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    print('Finished Training')

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')

    torch.save(model.state_dict(), 'models/vim_model.pth')
    print('Modelo salvo com sucesso!')
