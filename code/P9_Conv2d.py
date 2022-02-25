import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset=dataset, batch_size=64)


class Modul(nn.Module):
    def __init__(self):
        super(Modul, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)

        return x


step = 0
model1 = Modul()
for data in dataloader:
    imgs, target = data
    writer.add_images("conv2d_input", imgs, step)
    output = model1(imgs)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("conv2d_output", output, step)
    step = step + 1

writer.close()
