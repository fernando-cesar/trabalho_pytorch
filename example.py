import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

"""
x = torch.Tensor(10).random_(0, 10)
print(torch.cuda.is_available())
x.to("cuda")
print(x)
"""

# criando tensores

# print(torch.ones((2, 3)))

# print(torch.full((2, 3), 3.141592))

# print(torch.empty((2, 3)))

# valores aleatorios a partir de uma distribuição normal entre 0 e 1
# print(torch.rand((2, 3)))

# valores aleatorios com media 0 e variancia 1 de uma distribuição normal
# print(torch.randn((2, 3)))

# print(torch.randint(10, 100, (2, 3)))

# print(torch.tensor([[1, 2, 3], [4, 5, 6]]))

# criando tensor a partir de outro tensor
"""
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(a.dtype)
print(a.shape)
print(torch.numel(a))
print('\n')
b = torch.ones_like(a)
print(b)
print(b.dtype)
print(b.shape)
print('\n')
print(a.new_full((2, 2), 3.0))
"""

# ponte entre numpy array e tensores
"""
a = np.ones((2, 3))
print(a)
b = torch.from_numpy(a)
print(b)
"""

# cálculo de gradiente em tensor Q = 3*a^3 - b^2, supondo a e b serem parâmetros de uma Rede Neural, e Q ser o erro.
"""
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
Q = 3*a**3 - b**2
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)
print(9*a**2 == a.grad)
print(-2*b == b.grad)
"""

# reorganizar dados nos tensores
"""
a = torch.Tensor([1, 2, 3, 4])
print(a.shape)
print(torch.reshape(a, (2, 2)))  # cria copia
a.resize_((2, 2))                # nao cria copia
print(a)
print(a.shape)
a = torch.Tensor([1, 2, 3, 4, 5, 6])
print(a.view((2, 3)))
print('\n')
a = torch.Tensor([[1, 2, 3], [4, 5, 6]])
b = torch.Tensor([4, 5, 6, 7, 8, 9])
print(b)
print(b.view_as(a))
"""

# modelo para uma rede neural de camada única e definição da função de perda para avaliar o modelo
"""
input_units = 10
output_units = 1
model = nn.Sequential(nn.Linear(input_units, output_units), nn.Sigmoid())
print(model)
loss_funct = nn.MSELoss()     # calcula um valor que estima a distância que uma saída está do valor destino
print(loss_funct)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # pesos e vieses do modelo; lr: taxa de aprendizado
"""

"""
input_units = 10
output_units = 1
model = nn.Sequential(nn.Linear(input_units, output_units), nn.Sigmoid())
loss_funct = nn.MSELoss()
x = torch.randn(20, 10)
y = torch.randint(0, 2, (20, 1)).type(torch.FloatTensor)
optimizer = optim.Adam(model.parameters(), lr=0.01)
losses = []
for i in range(20):
    y_pred = model(x)              # chamado do modelo parsa executar a predição
    loss = loss_funct(y_pred, y)   # calculo da função de perda baseado em y_pred e y
    losses.append(loss.item())
    optimizer.zero_grad()          # gradientes são zerados
    loss.backward()                # calculo do gradiente da função de perda
    optimizer.step()               # atualiza os parametros
    if i % 5 == 0:
        print(i, loss.item())
plt.plot(range(0, 20), losses)
plt.show()
"""