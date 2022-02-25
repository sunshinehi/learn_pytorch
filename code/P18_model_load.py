import torch
import torchvision.models

# 方式1 ->加载模型

model = torch.load("vgg16_1.pth")
print(model)

# 方式2 ->加载模型，只有权重

vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_2.pth"))
print(vgg16)
