import torch
import torch.nn as nn
from torchvision import models

import deep_twist.data.utils as utils


def softmax_l1_loss(output, pos, lambda1=1.0):
    rect = pos[0]
    theta, rect_coords = output
    softmax = nn.CrossEntropyLoss()
    l1 = nn.SmoothL1Loss()
    theta_target = utils.discretize_theta(rect[:, 2], ticks=20)
    rect_target = torch.cat((rect[:, :2], rect[:, -2:]), 1)
    print(softmax(theta, theta_target)) 
    print(l1(rect_coords, rect_target))
    return softmax(theta, theta_target) + lambda1 * l1(rect_coords, rect_target)


def l1_loss(output, pos):
    rect_coords = output
    rect = pos[0]
    l1 = nn.SmoothL1Loss()
    return l1(rect_coords, rect)
 

def softmax_l2_loss(output, pos, lambda1=0.001):
    rect = pos[0]
    theta, rect_coords = output
    softmax = nn.CrossEntropyLoss()
    l2 = nn.MSELoss()
    theta_target = utils.discretize_theta(rect[:, 2], ticks=20)
    rect_target = torch.cat((rect[:, :2], rect[:, -2:]), 1)
    return softmax(theta, theta_target) + lambda1 * l2(rect_coords, rect_target)



class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.model_ft = nn.Sequential(*list(resnet.children())[:-1])
        self.theta_out = nn.Linear(resnet.fc.in_features, 20)
        self.box_fc = nn.Linear(resnet.fc.in_features, 80)
        self.box_out = nn.Linear(80, 4)
        
    def forward(self, x):
        ft_out = self.model_ft(x)
        ft_out = ft_out.view(ft_out.size(0), -1)
        theta = self.theta_out(ft_out)
        x = self.box_fc(ft_out)
        rect_coords = self.box_out(x)
        return (theta, rect_coords)


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.model_ft = nn.Sequential(*list(vgg16.features.children())[:-1])
        self.theta_out = nn.Linear(100352, 20)
        self.box_fc = nn.Linear(100352, 80)
        self.box_out = nn.Linear(80, 4)
        
    def forward(self, x):
        ft_out = self.model_ft(x)
        ft_out = ft_out.view(ft_out.size(0), -1)
        theta = self.theta_out(ft_out)
        x = self.box_fc(ft_out)
        rect_coords = self.box_out(x)
        return (theta, rect_coords)


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        alexnet = models.alexnet(pretrained=True)
        self.model_ft = nn.Sequential(*list(alexnet.features.children())[:-1])
        self.theta_out = nn.Linear(43264, 20)
        self.box_fc = nn.Linear(43264, 80)
        self.box_out = nn.Linear(80, 4)
        
    def forward(self, x):
        ft_out = self.model_ft(x)
        ft_out = ft_out.view(ft_out.size(0), -1)
        theta = self.theta_out(ft_out)
        x = self.box_fc(ft_out)
        rect_coords = self.box_out(x)
        return theta, rect_coords

class AlexNetPrime(nn.Module):
    def __init__(self):
        super(AlexNetPrime, self).__init__()
        alexnet = models.alexnet(pretrained=True)
        self.model_ft = nn.Sequential(*list(alexnet.features.children())[:-1])
        self.box_fc = nn.Linear(43264, 80)
        self.box_out = nn.Linear(80, 5)

    def forward(self, x):
        ft_out = self.model_ft(x)
        ft_out = ft_out.view(ft_out.size(0), -1)
        x = self.box_fc(ft_out)
        rects = self.box_out(x)
        return rects
