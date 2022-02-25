import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)
# 1保存方式
torch.save(vgg16, "vgg16_1.pth")

# 2保存方式（官方推荐）
torch.save(vgg16.state_dict(), "vgg16_2.pth")
