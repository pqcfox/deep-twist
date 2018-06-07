import numpy as np
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
    def __init__(self, boxes=[]):
        super(DeepGrasp, self).__init__()
        self.sizes = sizes
        self.aspects = aspects
        self.n = len(sizes) * len(aspects)

        resnet = models.resnet50(pretrained=True)
        self.model_ft = nn.Sequential(*list(resnet.children())[:7])
        self.rpn_conv = nn.Conv2d(1024, 512, kernel_size=(3, 3), padding=(1, 1))
        self.rpn_cls_conv = nn.Conv2d(512, 2 * self.n, kernel_size=(1, 1))
        self.rpn_reg_conv = nn.Conv2d(512, 4 * self.n, kernel_size=(1, 1))
        self.rcnn = nn.Sequential(*list(resnet.children())[7:9])
        self.theta_out = nn.Linear(resnet.fc.in_features, 20)
        self.box_fc = nn.Linear(resnet.fc.in_features, 80)
        self.box_out = nn.Linear(80, 4)

    def decode_anchor(self, x, y, index):
        size = self.sizes[index // 3]
        aspect = self.aspects[index % 3]
        return x, y, size * np.sqrt(aspect), size / np.sqrt(aspect)
        
    def get_proposals(self, rpn_cls, rpn_reg):
        is_grasp = rpn_cls[:, ::2, :, :] < rpn_cls[:, 1::2, :, :]
        proposals = []
        for i in range(rpn_cls.size(0)):
            ex_props = []
            grasp_index = is_grasp[i, :, :, :].nonzero().detach().numpy()
            for anchor_idx, y_base, x_base in grasp_index:
                x, y, w, h = self.decode_anchor(x_base, y_base, anchor_idx)
                dx, dy, dw, dh = rpn_reg[i, (4 * anchor_idx):(4 * anchor_idx + 4), y_base, x_base].detach().numpy()
                ex_props.append(torch.FloatTensor((x + dx, y + dy, w + dw, h + dh)))
            proposals.append(ex_props)
        return proposals

    def forward(self, x):
        ft_out = self.model_ft(x)
        rpn_ft = self.rpn_conv(ft_out)
        rpn_cls = self.rpn_cls_conv(rpn_ft)
        rpn_reg = self.rpn_reg_conv(rpn_ft)
        proposals = self.get_proposals(rpn_cls, rpn_reg)
        for img_props in proposals:
            for prop in img_props:
                # ROI pooling right here...
                rcnn_out = self.rcnn(roi_pooled)
                rcnn_flat = rcnn_out.view(rcnn_out.size(0), -1)
                theta = self.theta_out(rcnn_flat)
                rect_ft = self.box_fc(ft_out)
                rect_coords = self.box_out(rect_ft)
        return (theta, rect_coords)


    def forward(self, x):
        ft_out = self.model_ft(x)
        ft_out = ft_out.view(ft_out.size(0), -1)
        theta = self.theta_out(ft_out)
        x = self.box_fc(ft_out)
        rect_coords = self.box_out(x)
        return (theta, rect_coords)
