from sklearn.metrics import f1_score, precision_score, recall_score
from vision_mamba_nih_crx8 import Vim, NIHChestXrayDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import torch


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Mesmas transformações utilizadas no treinamento
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Carregar o dataset de teste
    test_dataset = NIHChestXrayDataset(
        csv_file='data/nih_cxr8/Data_Entry_2017.csv',
        root_dir='data/nih_cxr8/images',
        transform=transform_test,
        file_list='data/nih_cxr8/test_list.txt'
    )

    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Carregar o modelo salvo
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

    model.load_state_dict(torch.load('models/vim_nih_chest_xray_model.pth', map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    # Coletar predições e rótulos verdadeiros
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()  # binariza as predições
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Cálculo de métricas multi-label
    # Strict accuracy: quantas amostras tiveram todas as classes corretas
    strict_accuracy = (all_preds == all_labels).all(axis=1).mean()

    # F1, Precisão e Recall (micro)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    precision_micro = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    recall_micro = recall_score(all_labels, all_preds, average='micro', zero_division=0)

    # F1, Precisão e Recall (macro)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    class_names = [
        "No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
        "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
        "Fibrosis", "Pleural_Thickening", "Hernia"
    ]

    print("Strict Accuracy:", strict_accuracy)
    print("Micro-F1:", f1_micro)
    print("Micro-Precision:", precision_micro)
    print("Micro-Recall:", recall_micro)
    print("Macro-F1:", f1_macro)
    print("Macro-Precision:", precision_macro)
    print("Macro-Recall:", recall_macro)

    # Métricas por classe
    for i, cls in enumerate(class_names):
        cls_f1 = f1_score(all_labels[:, i], all_preds[:, i], average='binary', zero_division=0)
        cls_precision = precision_score(all_labels[:, i], all_preds[:, i], average='binary', zero_division=0)
        cls_recall = recall_score(all_labels[:, i], all_preds[:, i], average='binary', zero_division=0)
        print(f"Classe {cls} - F1: {cls_f1:.4f}, Precisão: {cls_precision:.4f}, Recall: {cls_recall:.4f}")
