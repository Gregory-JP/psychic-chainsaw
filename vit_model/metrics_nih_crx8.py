from vit_model_nih_crx8 import VisionTransformer, NIHChestXrayDataset
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import torch


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # As mesmas transformações usadas no treinamento para o conjunto de teste
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Definir caminhos e carregar dataset de teste
    csv_file = 'data/nih_cxr8/Data_Entry_2017.csv'
    root_dir = 'data/nih_cxr8/images'
    test_list_path = 'data/nih_cxr8/test_list.txt'

    test_dataset = NIHChestXrayDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        transform=transform_val,
        file_list=test_list_path
    )

    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Carregar o modelo
    model = VisionTransformer(num_classes=15).to(device)
    model.load_state_dict(torch.load('models/vision_transformer_nih_chest_xray.pth', map_location=device, weights_only=True))
    model.eval()

    # Vetores para armazenar predições e rótulos
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Aplicar sigmoid para obter probabilidades
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.1).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    strict_accuracy = (all_preds == all_labels).all(axis=1).mean()

    precision = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)

    f1_macro = f1_score(all_labels, all_preds, average='macro')
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')

    print("Strict Accuracy (todas as classes corretas por amostra):", strict_accuracy)
    print("Micro-F1:", f1)
    print("Micro-Precision:", precision)
    print("Micro-Recall:", recall)
    print("Macro-F1:", f1_macro)
    print("Macro-Precision:", precision_macro)
    print("Macro-Recall:", recall_macro)


    class_names = [
        "No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
        "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
        "Fibrosis", "Pleural_Thickening", "Hernia"
    ]

    # Report por classe
    for i, cls in enumerate(class_names):
        cls_precision = precision_score(all_labels[:, i], all_preds[:, i], average='binary', zero_division=0)
        cls_recall = recall_score(all_labels[:, i], all_preds[:, i], average='binary', zero_division=0)
        cls_f1 = f1_score(all_labels[:, i], all_preds[:, i], average='binary', zero_division=0)
        
        print(f"Class: {cls} - F1: {cls_f1:.4f}, Precision: {cls_precision:.4f}, Recall: {cls_recall:.4f}")
