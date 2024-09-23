from torchmetrics.classification import Accuracy

class MetricTracker:
    def __init__(self, device, task="multiclass", num_classes=10):
        # Inicializa as métricas de acurácia com parâmetros dinâmicos
        self.train_accuracy = Accuracy(task=task, num_classes=num_classes).to(device)
        self.val_accuracy = Accuracy(task=task, num_classes=num_classes).to(device)
        self.device = device

    def reset(self):
        # Reseta as métricas ao início de cada época
        self.train_accuracy.reset()
        self.val_accuracy.reset()

    def update_train(self, outputs, labels):
        # Atualiza a acurácia de treinamento
        self.train_accuracy.update(outputs.softmax(dim=-1), labels)

    def update_val(self, outputs, labels):
        # Atualiza a acurácia de validação
        self.val_accuracy.update(outputs.softmax(dim=-1), labels)

    def compute_train(self):
        # Computa a acurácia final de treinamento
        return self.train_accuracy.compute()

    def compute_val(self):
        # Computa a acurácia final de validação
        return self.val_accuracy.compute()
