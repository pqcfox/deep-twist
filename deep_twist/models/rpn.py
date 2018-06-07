import torch
import torch.nn as nn
from model.faster_rcnn.faster_rcnn import _fasterRCNN
from torchvision import models

import deep_twist.data.utils as utils


class DeepGrasp(_fasterRCNN):
    def __init__(self):
        super(DeepGrasp, self).__init__(classes=20, class_agnostic=True)
        self.dout_base_model = 1024

    def _init_modules(self):
        resnet = models.resnet50(pretrained=True)
        self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                       resnet.maxpool, resnet.layer1,
                                       resnet.layer2, resnet.layer3)
        self.RCNN_cls_score = nn.Linear(512, 2)
        self.RCNN_bbox_pred = nn.Linear(512, 4)
        self.RCNN_top = nn.Sequential(resnet.layer4, resnet.avgpool)

    def train(self, mode=True):
        nn.Module.train(self, mode)
        
    def _head_to_tail(self, pool5):
        fc7 = self.RCNN_top(pool5).mean(3).mean(2)

    def forward(self, x):
        ft_out = self.model_ft(x)
        ft_out = ft_out.view(ft_out.size(0), -1)
        theta = self.theta_out(ft_out)
        x = self.box_fc(ft_out)
        rect_coords = self.box_out(x)
        return (theta, rect_coords)
