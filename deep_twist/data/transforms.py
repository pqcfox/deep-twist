import torch

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
