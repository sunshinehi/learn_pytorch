import torchvision
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)

# print(test_set[0])
# print(test_set.classes)
#
# img, target = test_set[0]
# print(img)
# print(target)
# img.show()

# print(test_set[0])
for i in range(10):
    writer.add_image("CIFAR10", test_set[i][0], i)

writer.close()
