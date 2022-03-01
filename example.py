import torch

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
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(a.dtype)
print(a.shape)
print('\n')
b = torch.ones_like(a)
print(b)
print(b.dtype)
print(b.shape)
print('\n')
print(a.new_full((2, 2), 3.0))
