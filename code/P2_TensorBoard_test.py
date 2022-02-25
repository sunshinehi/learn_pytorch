from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")

# add_image事例
image_path = "data/hymenoptera_data/train/ants/0013035.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
# np.transpose将H W C转换为C H W
img_array1 = np.transpose(img_array)
writer.add_image("img1", img_array1, 1)

# add_scalar事例
for i in range(100):
    writer.add_scalar("y=2x", 2 * i, i)

writer.close()

# 终端打开localhost:6006
# tensorboard --logdir=logs
