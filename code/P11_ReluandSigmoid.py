import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import ReLU, Sigmoid


writer = SummaryWriter("logs")

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        output = self.sigmoid(x)
        return output


step = 0
model1 = Model()
for data in dataloader:
    imgs, target = data
    writer.add_images("Sigmoid_input", imgs, step)
    output = model1(imgs)
    writer.add_images("Sigmoid_output", output, step)
    step = step + 1

writer.close()
