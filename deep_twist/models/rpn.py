import torch
import torch.nn as nn
from model.faster_rcnn.faster_rcnn import _fasterRCNN
from torchvision import models

import deep_twist.data.utils as utils


class DeepGrasp(_fasterRCNN):
    def __init__(self):
        angles = ['angle_{}'.format(i) for i in range(19)] + ['no_angle']
        self.dout_base_model = 1024
        super(DeepGrasp, self).__init__(classes=angles, class_agnostic=True)

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
