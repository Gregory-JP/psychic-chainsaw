from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from vit_model_chest_xray import VisionTransformer, ChestXrayDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if __name__ == '__main__':
    test_dataset = ChestXrayDataset(root_dir='data/chest_xray/test', transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Carregar o modelo salvo
    model = VisionTransformer(num_classes=2).to(device)
    model.load_state_dict(torch.load('models/vision_transformer_chest_xray.pth', map_location=device, weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Armazenar as predições e rótulos verdadeiros
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Cálculo das métricas
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    print("Accuracy:", accuracy)
    print("F1-Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)

    print("\nClassification Report:\n", classification_report(all_labels, all_preds))
    print("\nConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))
