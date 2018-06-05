import torchvision
from deep_twist.data import utils, dataset, transforms
from skimage import io


def train_model(args, model, device, train_loader, dev_loader, optimizer, args):
    model.train()
    for epoch in range(args.epochs):
        for i, (data, target) in enumerate(train_loader):
            pass

        
        pass
        # get val loss
        pass
