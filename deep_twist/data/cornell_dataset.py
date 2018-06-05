import os
import re
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io

import dataset_utils


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
        depth = dataset_utils.parse_depth(depth_path, rgb.shape[:2])
        pos_path = os.path.join(self.root_dir, 'pcd{}cpos.txt'.format(id))
        pos = dataset_utils.parse_rects(pos_path, id)

        if self.transform:
            rgb = transforms.functional.to_pil_image(rgb)
            rgb, depth, pos = self.transform((rgb, depth, pos))
            rgb = transforms.functional.to_tensor(rgb)
        return rgb, depth, pos
