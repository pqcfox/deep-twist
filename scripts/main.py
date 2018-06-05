import sys
import matplotlib.pyplot as plt
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
from skimage import io

from deep_twist.data import dataset, transforms, utils


train_dataset = dataset.CornellGraspDataset(root_dir='cornell',
        transform=transforms.RandomRotate(0, 360))

rgb, depth, pos = train_dataset[0]
io.imsave('rgb.png', utils.draw_rectangle(rgb, pos[0]))
plt.imsave('depth.png', depth, cmap=plt.cm.jet)
