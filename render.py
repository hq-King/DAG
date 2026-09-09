import logging
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.DAG import get_DAG
from utils.loss import HM_Loss, kl_div
from utils.eval import evaluating, SIM
from data_utils.dataset_PIAD import PIAD
from sklearn.metrics import roc_auc_score
import numpy as np
import os
import pdb
import logging
import random
import yaml

from tools.utils.loss import L_ca
from tools.utils.evaluation import evaluate

import tensorboard
from tensorboardX import SummaryWriter

train_dataset = PIAD('train', Setting, point_train_path, img_train_path, box_train_path, 2)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8 ,shuffle=True, drop_last=True)

for i,(aff_name, img, points, labels, logits_labels, sub_box, obj_box) in enumerate(train_loader):

    for point, label, logits_label in zip(points, labels, logits_labels):
        affordance_gt = label.unsqueeze(dim=-1)
        Points = torch.from_numpy(point)
        Points = torch.unsqueeze(Points, 0)
        Points = Points.float().cuda()


        gt_point = o3d.geometry.PointCloud()
        gt_point.points = o3d.utility.Vector3dVector(point)

        color = np.zeros((2048,3))
        reference_color = np.array([255, 0, 0])
        back_color = np.array([190, 190, 190])

        for i, point_affordacne in enumerate(label):
            scale_i = point_affordacne
            color[i] = (reference_color-back_color) * scale_i + back_color
        gt_point.colors = o3d.utility.Vector3dVector(color.astype(np.float64) / 255.0)