import torch
import numpy as np
from skimage import transform


class RandomRotate(object):
    def __init__(self, min_angle, max_angle):
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.range = self.max_angle - self.min_angle

    def __call__(self, sample):
        rgb, depth, pos = sample
        angle = np.random.rand() * self.range + self.min_angle
        angle_rad = np.radians(angle)
        new_rgb = transform.rotate(rgb, angle, preserve_range=True).astype('uint8')
        new_depth = transform.rotate(depth, angle, preserve_range=True)
        new_pos = []
        x_center = rgb.shape[1] / 2
        y_center = rgb.shape[0] / 2
        for rect in pos[:1]:
            x_prime, y_prime = rect[0] - x_center, rect[1] - y_center
            x_rot = int(np.cos(-angle_rad) * x_prime - np.sin(-angle_rad) * y_prime)
            y_rot = int(np.sin(-angle_rad) * x_prime + np.cos(-angle_rad) * y_prime)
            new_pos.append((x_rot + x_center,
                            y_rot + y_center,
                            (rect[2] - angle) % 180,
                            rect[3],
                            rect[4]))
            
        return new_rgb, new_depth, new_pos


class RandomTranslate(object):
    def __init__(self, shift):
        self.shift = shift

    def __call__(self, sample):
        rgb, depth, pos = sample
        x_shift = np.random.randint(-self.shift, self.shift)
        y_shift = np.random.randint(-self.shift, self.shift)
        shift = (x_shift, y_shift)
        st = transform.SimilarityTransform(translation=shift)
        new_rgb = transform.warp(rgb, st, preserve_range=True).astype('uint8')
        new_depth = transform.warp(depth, st, preserve_range=True)
        new_pos = []
        for rect in pos:
            new_pos.append((rect[0] - x_shift,
                            rect[1] - y_shift,
                            rect[2],
                            rect[3],
                            rect[4]))
        return new_rgb, new_depth, new_pos


class Resize(object):
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, sample):
        rgb, depth, pos = sample
        h, w = rgb.shape[:2]
        new_rgb = transform.resize(rgb, (self.new_size, self.new_size), 
                preserve_range=True, mode='constant').astype('uint8')
        new_depth = transform.resize(depth, (self.new_size, self.new_size),
                preserve_range=True, mode='constant')
        new_pos = []
        for rect in pos:
            new_pos.append((rect[0] * self.new_size/w,
                            rect[1] * self.new_size/h,
                            rect[2],
                            rect[3] * self.new_size/w,
                            rect[4] * self.new_size/h))
        return new_rgb, new_depth, new_pos


class CenterCrop(object):
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, sample):
        rgb, depth, pos = sample
        h, w = rgb.shape[:2]
        i = (h - self.new_size)//2
        j = (w - self.new_size)//2
        new_rgb = rgb[i:i+self.new_size, j:j+self.new_size]
        new_depth = depth[i:i+self.new_size, j:j+self.new_size]
        new_pos = []
        for rect in pos:
            new_pos.append((rect[0] - j, rect[1] - i, rect[2], rect[3], rect[4]))
        return new_rgb, new_depth, new_pos


class ConvertToRGD(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        rgb, depth, pos = sample
        closest, furthest = np.min(depth), np.max(depth)
        depth -= closest
        depth *= 255 / (furthest - closest)
        rgb[:, :, 2] = depth
        return rgb, depth, pos


class SelectRandomPos(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        rgb, depth, pos = sample
        idx = np.random.choice(len(pos))
        return rgb, depth, [pos[idx]]


class SubtractImage(object):
    def __init__(self, mean):
        self.mean = mean 

    def __call__(self, sample):
        rgb, depth, pos = sample
        return rgb - self.mean, depth, pos
