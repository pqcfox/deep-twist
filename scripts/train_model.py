import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms 

import torch.optim
from torch.utils.data import DataLoader
from skimage import io

import deep_twist.models.baseline
from deep_twist.data import dataset, transforms
from deep_twist.data import utils as data_utils
from deep_twist.train import utils as train_utils 

parser = argparse.ArgumentParser(description='DeepTwist Grasp Detection Network')
parser.add_argument('--batch-size', type=int, default=32, 
                    help='batch size for training and validation') 
parser.add_argument('--epochs', type=int, default=3, help='number of epochs to train') 
parser.add_argument('--lr', type=float, default=0.01, help='learning rate') 
parser.add_argument('--log-interval', type=int, default=1, help='batches between logs')
parser.add_argument('--model', nargs='?', type=str, default='random', help='model to train') 
parser.add_argument('--val-interval', type=int, default=1, help='epochs between validations')
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
    val_transform= torchvision.transforms.Compose([transforms.ConvertToRGD(),
                                                   transforms.SubtractImage(144),
                                                   transforms.CenterCrop(321),
                                                   transforms.Resize(224),
                                                   transforms.SelectRandomPos()])

    train_dataset = dataset.CornellGraspDataset('cornell/train', transform=train_transform)
    val_dataset = dataset.CornellGraspDataset('cornell/val', transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    
    if args.model == 'random':
        model = deep_twist.models.baseline.Simple()
        loss = deep_twist.models.baseline.softmax_l2_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_utils.train_model(args, model, loss, train_loader, val_loader, optimizer)
    

if __name__ == '__main__':
    main()
