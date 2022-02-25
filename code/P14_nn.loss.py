import torch
from torch import nn

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targes = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targes = torch.reshape(targes, (1, 1, 1, 3))

loss = nn.L1Loss()
result = loss(inputs, targes)
print(result)

loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs, targes)
print(result_mse)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)
