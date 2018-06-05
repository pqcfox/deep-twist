import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary


class Simple(nn.Module):
    def __init__(self):
        super(Simple, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.model_ft = nn.Sequential(*list(resnet.children())[:-1])
        self.theta_fc = nn.Linear(resnet.fc.in_features, 20)
        self.theta_out = nn.Softmax()
        self.box_fc = nn.Linear(resnet.fc.in_features, 80)
        self.box_out = nn.Linear(80, 4)
        

    def forward(self, x):
        ft_out = self.model_ft(x)
        x = self.theta_fc(ft_out)
        theta = theta_out(x)
        x = self.box_fc(ft_out)
        x, y, w, h = self.box_out(x)
        return torch.FloatTensor((x, y, theta, w, h))
