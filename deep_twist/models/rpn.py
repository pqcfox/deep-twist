import torch
import torch.nn as nn
from torchvision import models

import deep_twist.data.utils as utils


def softmax_l2_loss(output, pos, lambda1=1.0):
    rect = pos[0]
    theta, rect_coords = output
    softmax = nn.CrossEntropyLoss()
    l2 = nn.MSELoss()
    theta_target = utils.discretize_theta(rect[:, 2], ticks=20)
    rect_target = torch.cat((rect[:, :2], rect[:, -2:]), 1)
    return softmax(theta, theta_target) + lambda1 * l2(rect_coords, rect_target)


class DeepGrasp(nn.Module):
    def __init__(self, k):
        super(DeepGrasp, self).__init__()
        resnet = resnet50(pretrained=True)
        self.model_ft = nn.Sequential(*list(resnet.children())[:7])
        self.rpn_conv = nn.Conv2d(1024, 512, kernel_size=(3, 3), padding=(1, 1))
        self.rpn_cls_conv = nn.Conv2d(512, 2 * k, kernel_size=(1, 1))
        self.rpn_reg_conv = nn.Conv2d(512, 4 * k, kernel_size=(1, 1))
        self.rcnn = nn.Sequential(*list(resnet.children())[7:9])
        self.theta_out = nn.Linear(resnet.fc.in_features, 20)
        self.box_fc = nn.Linear(resnet.fc.in_features, 80)
        self.box_out = nn.Linear(80, 4)

    def get_proposals(self, rpn_cls, rpn_reg):
        print(rpn_cls[::2] < rpn_cls[1::2])

    def forward(self, x):
        ft_out = self.model_ft(x)
        rpn_ft = self.rpn_conv(ft_out)
        rpn_cls = self.rpn_cls_conv(rpn_ft)
        rpn_reg = self.rpn_reg_conv(rpn_ft)
        proposals = self.get_proposals(rpn_cls, rpn_reg)
        # ROI pooling right here...
        rcnn_out = self.rcnn(roi_pooled)
        rcnn_flat = rcnn_out.view(rcnn_out.size(0), -1)
        theta = self.theta_out(rcnn_flat)
        rect_ft = self.box_fc(ft_out)
        rect_coords = self.box_out(rect_ft)
        return (theta, rect_coords)


"""
    def forward(self, x):
        ft_out = self.model_ft(x)
        ft_out = ft_out.view(ft_out.size(0), -1)
        theta = self.theta_out(ft_out)
        x = self.box_fc(ft_out)
        rect_coords = self.box_out(x)
        return (theta, rect_coords)
"""
