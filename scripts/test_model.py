import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms 

import torch.optim
from torch.utils.data import DataLoader

import deep_twist.models.baseline
from deep_twist.data import dataset, transforms
from deep_twist.evaluate import utils as eval_utils

parser = argparse.ArgumentParser(description='DeepTwist Grasp Detection Network')
parser.add_argument('--batch-size', type=int, default=32, 
                    help='batch size for training and validation') 
parser.add_argument('--device', nargs='?', type=str, default='cpu', help='device to use') 
parser.add_argument('--model', nargs='?', type=str, default='simple', help='model to train') 
parser.add_argument('--model_file', nargs='?', type=str, default='best_model.pt', 
                    help='model to evaluate') 
parser.add_argument('--use-val', dest='use_val', action='store_true',
                    help='use evaluation instead of test data') 
parser.set_defaults(use_val=False)

args = parser.parse_args()


def main():
    transform = torchvision.transforms.Compose([transforms.ConvertToRGD(),
                                                transforms.SubtractImage(144),
                                                transforms.CenterCrop(321),
                                                transforms.Resize(224)]) 

    path = 'cornell/val' if args.use_val else 'cornell/test'
    test_dataset = dataset.CornellGraspDataset(path, transform=transform)
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    model = torch.load(args.model_file)

    acc = eval_utils.eval_model(args, model, loader, progress=True)
    print('[{}] Acc: {}'.format('VAL' if args.use_val else 'TEST', acc))
    

if __name__ == '__main__':
    main()
