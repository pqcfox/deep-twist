import torchvision
from deep_twist.data import utils, dataset, transforms
from skimage import io


def train_model(args, model, loss, train_loader, val_loader, optimizer):
    model.train()
    for epoch in range(args.epochs):
        for i, (data, _, target) in enumerate(train_loader):
            print(loss(model(data), target))
            print('-----')
