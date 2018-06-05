import sys
import matplotlib.pyplot as plt
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
from skimage import io

from deep_twist.data import dataset, transforms, utils


train_dataset = dataset.CornellGraspDataset(root_dir='cornell', transform=transforms.CenterCrop(351))

print(train_dataset[0][2][0])
io.imsave('rgb.png', utils.draw_rectangle(train_dataset[0][0],
    train_dataset[0][2][0]))
plt.imsave('depth.png', train_dataset[0][1], cmap=plt.cm.jet)
