# Psychic Chainsaw

Este repositório contém o código e os arquivos relacionados ao meu trabalho final de curso sobre segmentação de imagens redes MAMBA (SSM).

## Estrutura do Repositório

```
- mamba_papers
- unet
    - data
    - model
    - main_model.py
    - unet-segmentation.pth
    - visualize_prediction.py
- LICENSE
- README
```

## Descrição dos Diretórios e Arquivos

- `mamba_papers/`: Contém os artigos e documentos relevantes sobre a arquitetura MAMBA.
- `unet`: Este diretório contém o código relacionado à implementação e ao treinamento do modelo U-Net.
- `data/`: Diretório onde os dados de treinamento e validação são armazenados (ignorado pelo Git).
- `main_model.py`: Script principal para treinar o modelo U-Net.
- `unet-segmentation.pth`: Arquivo do modelo treinado.
- `visualize_prediction.py`: Script para visualizar as predições do modelo treinado.

## Como Utilizar

### Pré-requisitos
- Python 3.8+
- Bibliotecas listadas no arquivo `requirements.txt` (por fazer)

### Instalação
1. Clone o repositório:
    ```bash
    git clone https://github.com/usuario/mamba_paper_repository.git
    cd mamba_paper_repository
    ```
2. Crie e ative um ambiente virtual:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    venv\Scripts\activate  # Windows
    ```
3. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

### Treinamento do Modelo
Para treinar o modelo U-Net, execute o script `main_model.py`:
```bash
python unet/main_model.py
```

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests para discutir melhorias e novas funcionalidades.

## Licença

Este projeto está licenciado sob os termos da licença MIT. Consulte o arquivo LICENSE para mais detalhes.

## Contato

Para qualquer dúvida ou sugestão, entre em contato através do email: pitthangregory@gmail.com