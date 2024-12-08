from sklearn.metrics import f1_score, precision_score, recall_score
from mamba_unet_nih_crx8 import MambaSSMUNet
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import torch
import os


class NIHChestXrayDataset(torch.utils.data.Dataset):
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

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # Mesmas transformações utilizadas no treinamento
    transform_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Dataset de teste
    test_dataset = NIHChestXrayDataset(
        csv_file='data/nih_cxr8/Data_Entry_2017.csv',
        root_dir='data/nih_cxr8/images',
        transform=transform_val,
        file_list='data/nih_cxr8/test_list.txt'
    )

    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Carregar o modelo
    model = MambaSSMUNet(num_classes=15).to(device)
    model.load_state_dict(torch.load('models/mamba_unet_nih_chest_xray.pth', map_location=device, weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []

    # Coletar predições e rótulos verdadeiros
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.1).float()  # Converte probabilidades em 0/1
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Cálculo de métricas multi-label
    # Strict accuracy (todas as classes corretas):
    strict_accuracy = (all_preds == all_labels).all(axis=1).mean()

    # F1, Precisão e Recall micro
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    precision_micro = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    recall_micro = recall_score(all_labels, all_preds, average='micro', zero_division=0)

    # F1, Precisão e Recall macro
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    print("Strict Accuracy:", strict_accuracy)
    print("Micro-F1:", f1_micro)
    print("Micro-Precision:", precision_micro)
    print("Micro-Recall:", recall_micro)
    print("Macro-F1:", f1_macro)
    print("Macro-Precision:", precision_macro)
    print("Macro-Recall:", recall_macro)

    class_names = [
        "No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
        "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
        "Fibrosis", "Pleural_Thickening", "Hernia"
    ]

    # Métricas por classe
    for i, cls in enumerate(class_names):
        cls_f1 = f1_score(all_labels[:, i], all_preds[:, i], average='binary', zero_division=0)
        cls_precision = precision_score(all_labels[:, i], all_preds[:, i], average='binary', zero_division=0)
        cls_recall = recall_score(all_labels[:, i], all_preds[:, i], average='binary', zero_division=0)
        print(f"Classe {cls} - F1: {cls_f1:.4f}, Precision: {cls_precision:.4f}, Recall: {cls_recall:.4f}")
