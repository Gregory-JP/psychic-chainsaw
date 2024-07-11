# Psychic Chainsaw

Este repositório contém o código e os arquivos relacionados ao meu trabalho final de curso sobre segmentação de imagens redes MAMBA (SSM).

## Vision Transformer and ResNet Training on CIFAR-10

Este projeto implementa um Vision Transformer e um modelo ResNet para treinar e avaliar a performance no dataset CIFAR-10. Várias técnicas de otimização e melhoramento de desempenho foram usadas para garantir uma melhor acurácia e eficiência no treinamento.

## Dataset

O dataset CIFAR-10 consiste em 60.000 imagens coloridas 32x32 em 10 classes, com 6.000 imagens por classe. Há 50.000 imagens de treinamento e 10.000 imagens de teste.

- **Classes**: avião, automóvel, pássaro, gato, cervo, cachorro, sapo, cavalo, navio, caminhão.

## Técnicas Utilizadas

### 1. **Data Augmentation**
Para aumentar a diversidade do dataset de treinamento e melhorar a generalização do modelo, foram aplicadas várias transformações nas imagens:

- Redimensionamento para 224x224 pixels.
- Flip horizontal aleatório.
- Rotação aleatória de até 10 graus.
- Normalização.

### 2. **Mixed Precision Training**
Para acelerar o treinamento e reduzir o uso de memória, foi utilizado o treinamento com precisão mista usando `torch.cuda.amp`.

### 3. **Scheduler para Taxa de Aprendizado**
Um scheduler foi utilizado para reduzir a taxa de aprendizado ao longo do tempo, ajudando o modelo a convergir mais suavemente:

```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
```

### 4. Modelos Utilizados
Vision Transformer

Um Vision Transformer foi inicialmente implementado, mas posteriormente substituído por um modelo ResNet18 pré-treinado para comparar a performance e eficiência.
Custom ResNet (ResNet18)

Um modelo ResNet18 pré-treinado foi utilizado e ajustado para classificar as 10 classes do CIFAR-10.

```python
class CustomResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomResNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
```

## Resultados e Visualização

Durante o treinamento, as métricas de perda (loss) e acurácia foram armazenadas e visualizadas ao final do processo.

## Como Executar

1. Clone o repositório
```
git clone <URL do Repositório>
```
```
cd <Nome do Repositório>
```

2. Instale as dependências
```
pip install -r requirements.txt
```

3. Execute o scipt
```python
python main.py
```

## Licença

Este projeto está licenciado sob os termos da licença MIT. Consulte o arquivo LICENSE para mais detalhes.

## Contato

Para qualquer dúvida ou sugestão, entre em contato através do email: pitthangregory@gmail.com