import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary


def softmax_l2_loss(output, pos):
    rect = pos[0]
    theta, rect_coords = output
    softmax = nn.CrossEntropyLoss()
    l2 = nn.MSELoss()
    theta_target = rect[:, 2].long()
    rect_target = torch.cat((rect[:, :2], rect[:, -2:]), 1)
    return softmax(theta, theta_target) + l2(rect_coords, rect_target)


class Simple(nn.Module):
    def __init__(self):
        super(Simple, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.model_ft = nn.Sequential(*list(resnet.children())[:-1])
        summary(self.model_ft, (3, 224, 224))
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


