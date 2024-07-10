import torch

print(torch.cuda.is_available())  # Deve retornar True se a GPU estiver disponível
print(torch.cuda.current_device())  # Deve retornar o índice da GPU atual, geralmente 0
print(torch.cuda.get_device_name(0))  # Deve retornar o nome da GPU, por exemplo, 'NVIDIA GeForce RTX 4060'
