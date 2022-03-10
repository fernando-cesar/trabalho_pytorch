"""
      Treinamento de rede neural convolucional
      para problema de classificação de imagens
"""

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time

if torch.cuda.device_count() >= 1:
    print("Let's use\n", torch.cuda.device_count(), "GPUs!")

inicio = time.time()

# Transformações nos dados, que serão convertidos em tensores, e normalização dos valores dos pixels
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Definição de um tamanho de lote de 100 imagens e download dos dados de treinamento e de teste do conjunto de dados
batch_size = 100
train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)

# Define amostras de dados de forma aleatória na proporção de 20% pra 80%
dev_size = 0.2
idx = list(range(len(train_data)))   # cria uma lista de indices com o tamanho dos dados de treinamento
np.random.shuffle(idx)               # embralha a lista
split_size = int(np.floor(dev_size * len(train_data)))   # calcula o ponto de divisão dos dados
train_idx, dev_idx = idx[split_size:], idx[:split_size]  # separa a lista de indices original em duas
train_sampler = SubsetRandomSampler(train_idx)  # cria amostra randômica da lista de indices p/ o conjunto de dados
dev_sampler = SubsetRandomSampler(dev_idx)

# Define-se os lotes de cada conjunto de dados para serem usados
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)  # dados de treino
dev_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=dev_sampler)      # dados validação
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)                           # dados de teste


# Definindo a arquitetura de um rede neural (camadas de rede)
class CNN(nn.Module):
    def __init__(self):

        super(CNN, self).__init__()
        # argumentos: canal entrada, filtro, tamanho de cada filtro, passo e preenchimento
        self.conv1 = nn.Conv2d(3, 10, 3, 1, 1)
        self.conv2 = nn.Conv2d(10, 20, 3, 1, 1)
        self.conv3 = nn.Conv2d(20, 40, 3, 1, 1)
        # argumentos: tamanho de cada filtro e passo
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(40 * 4 * 4, 100)  # Uma camada totalmente conectada que gera uma saída de 100 unidades
        self.linear2 = nn.Linear(100, 10)  # camada totalmente conectada que gera 10 saídas
        # 20% de chance de um neuronio ser zerado, simulando treinamento em diferentes arquiteturas; reduz overfitting
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):    # define os cálculos a serem realizados nas informações conforme ele passa pelas camadas
        # usa a função de ativação ReLU após cada camada convolucional
        # usa a camada de pooling após cada camada convolucional para simplificando informaçãoes
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 40 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = F.log_softmax(self.linear2(x), dim=1)  # função de ativação para a camada de saída
        return x


model = CNN().to("cuda")
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 50  # número de iterações

train_losses, dev_losses, train_acc, dev_acc = [], [], [], []
x_axis = []
# loop para cada iteração
for e in range(1, epochs+1):
    losses = 0
    acc = 0
    iterations = 0
    model.train()

    # treinamento
    for data, target in train_loader:
        iterations += 1
        # propagação e retropropagação
        pred = model(data.to("cuda"))
        loss = loss_function(pred, target.to("cuda"))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.item()
        p = torch.exp(pred)
        top_p, top_class = p.topk(1, dim=1)
        acc += accuracy_score(target.to("cpu"), top_class.to("cpu"))
    dev_losss = 0
    dev_accs = 0
    iter_2 = 0
    # Validação do modelo numa dada iteração
    if e % 5 == 0 or e == 1:
        x_axis.append(e)
        with torch.no_grad():
            model.eval()
            # loop sobre os dados de validação
            for data_dev, target_dev in dev_loader:
                iter_2 += 1
                dev_pred = model(data_dev.to("cuda"))
                dev_loss = loss_function(dev_pred, target_dev.to("cuda"))
                dev_losss += dev_loss.item()
                dev_p = torch.exp(dev_pred)
                top_p, dev_top_class = dev_p.topk(1, dim=1)
                dev_accs += accuracy_score(target_dev.to("cpu"), dev_top_class.to("cpu"))
        # perdas e precisões anexadas para serem exibidas
        train_losses.append(losses / iterations)
        dev_losses.append(dev_losss / iter_2)
        train_acc.append(acc / iterations)
        dev_acc.append(dev_accs / iter_2)
        print("Epoch: {}/{}.. ".format(e, epochs), "Training Loss: {:.3f}.. ".format(losses / iterations),
              "Validation Loss: {:.3f}.. ".format(dev_losss / iter_2),
              "Training Accuracy: {:.3f}.. ".format(acc / iterations),
              "Validation Accuracy: {:.3f}".format(dev_accs / iter_2))

model.eval()
iter_3 = 0
acc_test = 0
for data_test, target_test in test_loader:
    iter_3 += 1
    test_pred = model(data_test.to("cuda"))
    test_pred = torch.exp(test_pred)
    top_p, top_class_test = test_pred.topk(1, dim=1)
    acc_test += accuracy_score(target_test .to("cpu"), top_class_test .to("cpu"))

print(f'\nTest Accuracy: {acc_test/iter_3}')

# após a 15ª iteração, o overfitting começa a afetar o modelo

# A precisão do conjunto de teste é muito semelhante à precisão alcançada pelos outros dois conjuntos,
# o que significa que o modelo tem a capacidade de funcionar igualmente bem em dados não vistos;
# Deve ser em torno de 72%

fim = time.time()
print(f'\nTempo de execucao {fim - inicio}')

plt.plot(x_axis, train_losses, label='Training loss')
plt.plot(x_axis, dev_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()
plt.plot(x_axis, train_acc, label="Training accuracy")
plt.plot(x_axis, dev_acc, label="Validation accuracy")
plt.legend(frameon=False)
plt.show()
