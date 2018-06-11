import sys
from os.path import dirname, realpath, join
sys.path.append(dirname(dirname(realpath(__file__))))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms 

import torch.optim
from torch.utils.data import DataLoader
from skimage import io

import deep_twist.models.baseline
import deep_twist.models.rpn
from deep_twist.data import dataset, transforms
from deep_twist.data import utils as data_utils
from deep_twist.train import utils as train_utils 

parser = argparse.ArgumentParser(description='DeepTwist Grasp Detection Network')
parser.add_argument('--batch-size', type=int, default=16, 
                    help='batch size for training and validation') 
parser.add_argument('--device', nargs='?', type=str, default='cpu', help='device to use') 
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train') 
parser.add_argument('--is-this-loss', type=str, default='l1', help='loss to use for theta')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate') 
parser.add_argument('--log-interval', type=int, default=1, help='batches between logs')
parser.add_argument('--model', nargs='?', type=str, default='alexnet', help='model to train') 
parser.add_argument('--val-interval', type=int, default=5, help='epochs between validations')
args = parser.parse_args()


def main():
    train_transform = torchvision.transforms.Compose([transforms.ConvertToRGD(),
                                                      transforms.SubtractImage(144),
                                                      transforms.CenterCrop(351),
                                                      transforms.RandomRotate(0, 360),
                                                      transforms.CenterCrop(321),
                                                      transforms.RandomTranslate(50),
                                                      transforms.Resize(224),
                                                      transforms.SelectRandomPos()])
    val_transform = torchvision.transforms.Compose([transforms.ConvertToRGD(),
                                                    transforms.SubtractImage(144),
                                                    transforms.CenterCrop(321),
                                                    transforms.Resize(224)]) 

    train_dataset = dataset.CornellGraspDataset('cornell/train', transform=train_transform)
    val_dataset = dataset.CornellGraspDataset('cornell/val', transform=val_transform)
    test_dataset = dataset.CornellGraspDataset('cornell/test', transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
            shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
            shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
            shuffle=False)
    
    if args.is_this_loss == 'l1':
        loss = deep_twist.models.baseline.l1_loss
    else:
        loss = deep_twist.models.baseline.softmax_l1_loss

    one_hot=True
    if args.model == 'resnet':
        model = deep_twist.models.baseline.ResNet()
    if args.model == 'alexnet':
        model = deep_twist.models.baseline.AlexNet()
        loss = deep_twist.models.baseline.softmax_l1_loss
    if args.model == 'alexnet_prime':
        model = deep_twist.models.baseline.AlexNetPrime()
        loss = deep_twist.models.baseline.l1_loss
        one_hot=False
    if args.model == 'vgg16':
        model = deep_twist.models.baseline.VGG16()


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_utils.train_model(args, model, loss, train_loader, val_loader,
            test_loader, optimizer, one_hot=one_hot)
    

if __name__ == '__main__':
    main()
