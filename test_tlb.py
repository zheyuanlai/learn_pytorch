from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter('logs')
image_path = 'hymenoptera_data/train/ants/0013035.jpg'
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

# images
writer.add_image('test', img_array, 1, dataformats='HWC')

# y=2x (scalar)
for i in range(100):
    writer.add_scalar('y=2x', 2 * i, i)

writer.close()