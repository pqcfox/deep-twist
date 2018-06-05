import os
import re
import numpy as np
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from skimage import io

from deep_twist.data import utils 


class CornellGraspDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform 
        self.ids = []
        for fname in os.listdir(self.root_dir):
            if fname.endswith('.png'):
                id = re.findall('\d+', fname)[0]
                self.ids.append(id)
        self.ids.sort()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        rgb_path = os.path.join(self.root_dir, 'pcd{}r.png'.format(id))
        rgb = io.imread(rgb_path)
        depth_path = os.path.join(self.root_dir, 'pcd{}.txt'.format(id))
        depth = utils.parse_depth(depth_path, rgb.shape[:2])
        pos_path = os.path.join(self.root_dir, 'pcd{}cpos.txt'.format(id))
        pos = utils.parse_rects(pos_path, id)

        if self.transform:
            depth_scale = np.max(depth)
            rgb, depth, pos = self.transform((rgb, depth, pos))
        return rgb, depth, pos


def load_dataset(root_dir, train_split=0.6, val_split=0.2, transform=None):
    dataset = CornellGraspDataset(root_dir=root_dir, transform=transform)
    train_count = int(train_split * len(dataset))
    val_count = int(val_split * len(dataset))
    test_count = len(dataset) - (train_count + val_count)
    return random_split(dataset, [train_count, val_count, test_count])
