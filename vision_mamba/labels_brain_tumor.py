import os

def count_labels(labels_dir):
    total_labels = 0
    count_0 = 0
    count_1 = 0

    # Iterar sobre os arquivos de rótulos
    for label_file in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, label_file)
        
        # Verificar se o arquivo é um arquivo de texto
        if os.path.isfile(label_path) and label_file.endswith('.txt'):
            with open(label_path, 'r') as f:
                line = f.readline().strip()
                if line:  # Verificar se a linha não está vazia
                    label = int(line.split()[0])  # Pegar apenas a classe (primeiro número)
                    total_labels += 1
                    if label == 0:
                        count_0 += 1
                    elif label == 1:
                        count_1 += 1

    return total_labels, count_0, count_1


# Diretório de rótulos
labels_dir = r'data\brain_tumor\valid\labels'  # Altere para o caminho do seu diretório de rótulos

# Contar os rótulos
total_labels, count_0, count_1 = count_labels(labels_dir)

# Exibir os resultados
print(f"Total de rótulos: {total_labels}")
print(f"Classe 0: {count_0}")
print(f"Classe 1: {count_1}")
