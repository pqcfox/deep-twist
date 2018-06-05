import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary


class Simple(nn.Module):
    def __init__(self):
        super(Simple, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.model_ft = nn.Sequential(*list(resnet.children())[:-1])
        summary(self.model_ft, (3, 224, 224))
        self.theta_fc = nn.Linear(resnet.fc.in_features, 20)
        self.theta_out = nn.Softmax()
        self.box_fc = nn.Linear(resnet.fc.in_features, 80)
        self.box_out = nn.Linear(80, 4)
        
    def forward(self, x):
        ft_out = self.model_ft(x)
        ft_out = ft_out.view(ft_out.size(0), -1)
        x = self.theta_fc(ft_out)
        theta = self.theta_out(x)
        x = self.box_fc(ft_out)
        rect_coords = self.box_out(x)
        return (theta, rect_coords)
