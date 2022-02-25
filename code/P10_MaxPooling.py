import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import MaxPool2d

# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)
#
# input = torch.reshape(input, (1, 1, 5, 5))
# print(input.shape)
writer = SummaryWriter("logs")

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        output = self.maxpool1(x)
        return output


step = 0
model1 = Model()
for data in dataloader:
    imgs, target = data
    writer.add_images("MaxPooling_input", imgs, step)
    output = model1(imgs)
    writer.add_images("MaxPooling_output", output, step)
    step = step + 1

writer.close()
