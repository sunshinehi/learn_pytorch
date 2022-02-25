import torch
import torchvision.datasets
from torch import nn, optim
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),
        )

    def forward(self, x):
        x = self.model1(x)
        return x


model1 = Model()
print(model1)

loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model1.parameters(), lr=0.01)

for epoch in range(20):
    total_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = model1(imgs)
        result_loss = loss(outputs, targets)
        optimizer.zero_grad()
        result_loss.backward()
        optimizer.step()
        total_loss = total_loss + result_loss
    print(total_loss)
