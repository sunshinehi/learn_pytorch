from torchvision import transforms
from PIL import Image

# import cv2

# python的用法 -> tensor数据类型
# transforms.ToTensor()
# 1.transforms该如何被使用(python)
# 2.为什么我们需要Tensor数据类型


# 绝对路径 D:\Desktop\learn_python\data\hymenoptera_data\train\ants\0013035.jpg
# 相对路径 data/hymenoptera_data/train/ants/0013035.jpg
img_path = "data/hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(img_path)
#  __call__()。该方法的功能类似于在类中重载 () 运算符，使得类实例对象可以像调用普通函数那样，以“对象名()”的形式使用。
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)
