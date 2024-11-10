import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, Tensor
from zeta.nn import SSM
from einops.layers.torch import Reduce
import wandb


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

# Configurações do Weights & Biases
def setup_wandb(project_name, run_name, config):
    wandb.init(project=project_name, name=run_name, config=config)

def log_metrics(epoch, train_loss, val_loss=None, val_accuracy=None):
    metrics = {
        'epoch': epoch,
        'train_loss': train_loss,
    }
    if val_loss is not None:
        metrics['val_loss'] = val_loss
    if val_accuracy is not None:
        metrics['val_accuracy'] = val_accuracy
    wandb.log(metrics)

if __name__ == '__main__':

    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    # from load_env import  wandb_key

    # api_key = wandb_key()

    # Configuração do projeto e dos hiperparâmetros para o W&B
    config = {
        'learning_rate': 0.001,
        'batch_size': 32,  # Ajuste o batch size para 32
        'num_epochs': 100,
        'optimizer': 'Adam',
        'loss_function': 'CrossEntropyLoss',
        'model': 'Vim',
        'dataset': 'Chest X-ray Pneumonia'
    }

    # wandb.login(key=api_key)
    wandb.login(key='')

    setup_wandb(project_name='Chest-Xray-Pneumonia', run_name='Vim_Training_Run', config=config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transformações para os dados de treinamento e validação
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # tamanho correspondente ao modelo? 
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

    # Carregar os datasets usando ImageFolder
    train_dataset = datasets.ImageFolder(root='data/chest_xray/train', transform=transform_train)
    val_dataset = datasets.ImageFolder(root='data/chest_xray/val', transform=transform_val)
    test_dataset = datasets.ImageFolder(root='data/chest_xray/test', transform=transform_val)

    # Dataloaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # Instanciar o modelo com num_classes=2
    model = Vim(
        dim=128,  
        dt_rank=16,
        dim_inner=128,
        d_state=128,
        num_classes=2,  # Ajustado para 2 classes
        image_size=224,
        patch_size=32,
        channels=3,
        dropout=0.1,
        depth=6  
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    num_epochs = config['num_epochs']

    print(f'Treinamento iniciado em {device}')
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")

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
        
        # Validação a cada época
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # Log do W&B
        log_metrics(epoch, train_loss=avg_loss, val_loss=avg_val_loss, val_accuracy=accuracy)

    print('Finished Training')

    # Avaliação no conjunto de teste
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    # Log final do W&B para teste
    wandb.log({'test_loss': avg_test_loss, 'test_accuracy': test_accuracy})

    torch.save(model.state_dict(), 'models/vim_chest_xray_model.pth')
    print('Modelo salvo com sucesso!')