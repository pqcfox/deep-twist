import os
import re
import numpy as np
import shapely.geometry

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from skimage import io, transform
from tqdm import tqdm

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


dataset = CornellGraspDataset('cornell')
print(dataset[0])


"""
class RandomTranslate(object):
    def __init__(self, shift):
        self.shift = shift

    def __call__(self, sample):
        img, gt = sample
        x_shift = np.random.randint(-self.shift, self.shift)
        y_shift = np.random.randint(-self.shift, self.shift)
        shift = (x_shift, y_shift)
        new_img = transforms.functional.affine(img, 0, shift, 1, 0)
        new_gt = np.copy(gt)
        new_gt[0] += x_shift
        new_gt[1] += y_shift
        return new_img, new_gt


class Resize(object):
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, sample):
        img, gt = sample
        h, w = img.size[:2]
        new_img = transforms.functional.resize(img, (self.new_size,
            self.new_size)) 
        new_gt = np.copy(gt)
        new_gt[0] *= self.new_size/w
        new_gt[1] *= self.new_size/h
        new_gt[3] *= self.new_size/w
        new_gt[4] *= self.new_size/h
        return new_img, new_gt


class CenterCrop(object):
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, sample):
        img, gt = sample
        h, w = img.size[:2]
        i = (h - self.new_size)//2
        j = (w - self.new_size)//2
        new_img = transforms.functional.crop(img, i, j, self.new_size, self.new_size) 
        new_gt = np.copy(gt)
        new_gt[0] -= j
        new_gt[1] -= i
        return new_img, new_gt


class GraspNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean)
        self.std = torch.FloatTensor(std)

    def __call__(self, sample):
        img, gt = sample
        return img, (gt - self.mean)/self.std


class ResNetGrasp(nn.Module):
    def __init__(self):
        super(ResNetGrasp, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])

        num_ftrs = self.resnet.fc.in_features
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Linear(128, 5))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
"""
