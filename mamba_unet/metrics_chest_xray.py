from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from mamba_unet_chest_xray import MambaSSMUNet
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch
import os


# Definir o dataset
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

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # Transformação (mesma usada no treino)
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Carregar dataset de teste
    test_dataset = ChestXrayDataset(root_dir='data/chest_xray/test', transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Carregar o modelo
    model = MambaSSMUNet(img_size=112, num_classes=2).to(device)
    model.load_state_dict(torch.load('models/mamba_unet_chest_xray.pth', map_location=device, weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calcular métricas
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
    # Nomes das classes
    class_names = ["Normal", "Pneumonia"]

    print("Accuracy:", accuracy)
    print("F1-Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)
    print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=class_names))
    print("\nConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))
