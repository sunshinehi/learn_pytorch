import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Linear

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = Linear(196608, 10)

    def forward(self, x):
        output = self.linear(x)
        return output


model1 = Model()
for data in dataloader:
    imgs, target = data
    imgs = torch.flatten(imgs)
    print(imgs.shape)
    output = model1(imgs)
    print(output.shape)
