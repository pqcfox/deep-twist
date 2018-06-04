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


class CornellGraspDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform 
        self.ids = []
        for fname in os.listdir(self.root_dir):
            if fname.endswith('.png'):
                id = re.findall('\d+', fname)[0]
                self.ids.append(id)
        self.ids.sort()

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def parse_depth(depth_path, depth_shape):
        depth = np.zeros(depth_shape)
        width = depth.shape[1]
        with open(depth_path) as f:
            for line in f.read().splitlines():
                if line[0].isdigit():
                    values = line.split(' ')
                    z, depth_idx = float(values[2]), int(values[4])
                    row, col = depth_idx // width, depth_idx % width
                    depth[row, col] = z
        return depth

    @staticmethod
    def parse_rects(rect_path, id):
        rects = [] 
        with open(rect_path) as f:
            lines = f.read().splitlines()

            for i in range(0, len(lines), 4):
                rect_lines = lines[i:(i + 4)]
                points = []

                valid = True
                for rect_line in rect_lines:
                    point_values = rect_line.strip().split(' ')
                    point = tuple(float(value) for value in point_values)
                    if np.any(np.isnan(point)):
                        valid = False
                    points.append(point)

                if not valid:
                    continue

                x, y = point_mid(points[0], points[2])
                theta = point_angle(points[1], points[2])
                w = point_dist(points[0], points[1])
                h = point_dist(points[1], points[2])

                rects.append(torch.FloatTensor([x, y, theta, w, h]))
                with open('derp.csv', 'a') as f:
                    f.write(','.join([str(v) for v in (x, y, theta, w, h)]) + '\n')

        return rects

    def __getitem__(self, idx):
        id = self.ids[idx]
        img_path = os.path.join(self.root_dir, 'pcd{}r.png'.format(id))
        img = io.imread(img_path)

        pos_path = os.path.join(self.root_dir, 'pcd{}cpos.txt'.format(id))
        pos = self.parse_rects(pos_path, id)
        gt = pos[np.random.choice(len(pos))]
        gt = pos[0]

        if self.transform:
            img = transforms.functional.to_pil_image(img)
            img, gt = self.transform((img, gt))
            img = transforms.functional.to_tensor(img)

        return img, gt 


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
