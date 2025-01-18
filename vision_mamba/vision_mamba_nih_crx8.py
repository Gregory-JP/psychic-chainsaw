import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import Tensor
from zeta.nn import SSM  # Ensure this module is accessible
import wandb

# Utility Functions
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# Modified Output Head for Multi-Label Classification
def output_head(dim: int, num_classes: int):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, num_classes),
    )

# Definição do modelo
class VisionEncoderMambaBlock(nn.Module):
    def __init__(self, dim, dt_rank, dim_inner, d_state):
        super().__init__()
        self.forward_conv1d = nn.Conv1d(dim, dim, 1)
        self.backward_conv1d = nn.Conv1d(dim, dim, 1)
        self.norm = nn.LayerNorm(dim)
        self.silu = nn.SiLU()
        self.ssm = SSM(dim, dt_rank, dim_inner, d_state)
        self.proj = nn.Linear(dim, dim)
        self.softplus = nn.Softplus()

    def forward(self, x):
        skip = x
        x = self.norm(x)
        z1 = self.proj(x)
        x1 = self.process_direction(x, self.forward_conv1d, self.ssm)
        x2 = self.process_direction(x, self.backward_conv1d, self.ssm)
        z = self.silu(z1)
        x1 *= z
        x2 *= z
        return x1 + x2 + skip

    def process_direction(self, x, conv1d, ssm):
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

        self.dropout_layer = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
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
        x = self.dropout_layer(x)
        for layer in self.layers:
            x = layer(x)
        # Remove the class token before the output head if necessary
        x = x[:, 1:, :]  # Remove the class token if not needed
        x = self.output_head(x.mean(dim=1))  # Mean over sequence length
        return x

# Weights & Biases Setup Functions
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

# NIHChestXrayDataset Class
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

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

# Main Training Script
if __name__ == '__main__':
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torch.optim.lr_scheduler import StepLR

    # WandB Configuration
    config = {
        'learning_rate': 3e-4,
        'batch_size': 32,
        'num_epochs': 50,
        'optimizer': 'Adam',
        'loss_function': 'BCEWithLogitsLoss',
        'model': 'Vim',
        'dataset': 'NIH Chest X-rays'
    }

    wandb.login(key='ac22f8f654233aa519a442a6947d44d2c44f72c8')

    setup_wandb(project_name='NIH-Chest-Xrays', run_name='Vim_Training_Run', config=config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Data Transformations
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjusted to match the model's expected input size
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),  # Same size as training data
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Dataset Paths
    train_dataset = NIHChestXrayDataset(
        csv_file='data/nih_cxr8/Data_Entry_2017.csv',
        root_dir='data/nih_cxr8/images',
        transform=transform_train,
        file_list='data/nih_cxr8/train_val_list.txt'
    )

    val_dataset = NIHChestXrayDataset(
        csv_file='data/nih_cxr8/Data_Entry_2017.csv',
        root_dir='data/nih_cxr8/images',
        transform=transform_val,
        file_list='data/nih_cxr8/train_val_list.txt'
    )

    test_dataset = NIHChestXrayDataset(
        csv_file='data/nih_cxr8/Data_Entry_2017.csv',
        root_dir='data/nih_cxr8/images',
        transform=transform_val,
        file_list='data/nih_cxr8/test_list.txt'
    )

    # Data Loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8, pin_memory=True)

    # Model Initialization
    model = Vim(
        dim=128,
        dt_rank=16,
        dim_inner=128,
        d_state=128,
        num_classes=15,
        image_size=224,
        patch_size=32,
        channels=3,
        dropout=0.1,
        depth=6
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    num_epochs = config['num_epochs']
    train_losses = []
    val_losses = []
    val_accuracies = []

    print(f'Training started on {device}')
    if device.type == 'cuda':
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
        train_losses.append(avg_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Predictions
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.numel()

        val_accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')

        # Log to WandB
        log_metrics(epoch, avg_loss, avg_val_loss, val_accuracy)
        scheduler.step()

    print('Finished Training')

    # Test Evaluation
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.numel()

    test_accuracy = 100 * correct / total
    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    # Log final test metrics to WandB
    wandb.log({'test_loss': avg_test_loss, 'test_accuracy': test_accuracy})

    torch.save(model.state_dict(), 'models/vim_nih_chest_xray_model.pth')
    print('Model saved successfully!')