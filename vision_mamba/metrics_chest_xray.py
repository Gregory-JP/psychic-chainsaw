from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vision_mamba_chest_xray import Vim
import numpy as np
import torch


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # Transformações iguais às usadas no treinamento/validação
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Dataset de teste, assumindo estrutura de pastas igual (NORMAL e PNEUMONIA)
    test_dataset = datasets.ImageFolder(root='data/chest_xray/test', transform=transform_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Carregar o modelo salvo
    model = Vim(
        dim=128,
        dt_rank=16,
        dim_inner=128,
        d_state=128,
        num_classes=2,
        image_size=224,
        patch_size=32,
        channels=3,
        dropout=0.1,
        depth=6
    ).to(device)

    model.load_state_dict(torch.load('models/vim_chest_xray_model.pth', map_location=device))
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

    # Converter para numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Cálculo das métricas
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    class_names = ['Normal', 'Pneumonia']

    print("Accuracy:", accuracy)
    print("F1-Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)
    print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=class_names))
    print("\nConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))
