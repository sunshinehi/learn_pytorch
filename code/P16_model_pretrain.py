import torchvision

# train_data = torchvision.datasets.ImageNet("../dataset", split='train', download=True)
from torch import nn

vgg_false = torchvision.models.vgg16(pretrained=False)
vgg_true = torchvision.models.vgg16(pretrained=True)

train_data = torchvision.datasets.CIFAR10(root="../dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

vgg_true.classifier.add_module('add_liner', nn.Linear(1000,10))
print(vgg_true)

vgg_false.classifier[6] = nn.Linear(4096,10)
print(vgg_false)