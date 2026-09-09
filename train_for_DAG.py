import os
import logging
os.makedirs("runs/train/DAG", exist_ok=True)
logging.basicConfig(filename="runs/train/DAG/train.log", level=logging.INFO)

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
import pdb
import logging
import random
import yaml

from tools.utils.loss import L_ca
from tools.utils.evaluation import evaluate

import tensorboard
from tensorboardX import SummaryWriter


def read_yaml(path):
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)

    return dict

def main(opt, dict):
    if opt.use_gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    save_path = opt.save_dir + opt.name
    foler = os.path.exists(save_path)
    if not foler:
        os.makedirs(save_path)

    loger = logging.getLogger('Training')
    loger.setLevel(logging.INFO)
    log_name = opt.save_dir + opt.name + '/' + opt.log_name

    writer = SummaryWriter('logs/')

    def log_string(str):
        loger.info(str)
        print(str)

    img_train_path = dict['img_train']
    point_train_path = dict['point_train']
    img_val_path = dict['img_test']
    point_val_path = dict['point_test']
    box_train_path = dict['box_train']
    box_val_path = dict['box_test']
    Setting = dict['Setting']
    batch_size = dict['batch_size']

    log_string('Start loading train data---')
    train_dataset = PIAD('train', Setting, point_train_path, img_train_path, box_train_path, dict['pairing_num'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8 ,shuffle=True, drop_last=True)
    log_string(f'train data loading finish, loading data files:{len(train_dataset)}')

    log_string('Start loading val data---')
    val_dataset = PIAD('val', Setting, point_val_path, img_val_path, box_val_path)
    test_num = len(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    log_string(f'val data loading finish, loading data files:{len(val_dataset)}')

    model = get_DAG(img_model_path=dict['res18_pre'], N_p=dict['N_p'], emb_dim=dict['emb_dim'],
                       proj_dim=dict['proj_dim'], num_heads=dict['num_heads'], N_raw=dict['N_raw'],
                       num_affordance = dict['num_affordance'])

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(pg, lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=dict['Epoch'], eta_min=1e-6)

    if opt.resume:
        model_checkpoint = torch.load(opt.checkpoint_path, map_location='cuda:0')
        model.cuda()
        model.load_state_dict(model_checkpoint['model'])
        optimizer.load_state_dict(model_checkpoint['optimizer'])
        start_epoch = model_checkpoint['Epoch']
    else:
        start_epoch = -1
    
    model = model.to(device)
    batches_train, batches_val = len(train_loader), len(val_loader)

    loss_ca = L_ca().to(device)

   

    # best_AUC = 0
    best_current = {'AUC':0, 'aIOU':0, 'SIM':0}
    total_iter = 0
    eval_step = 100
    show_step = 10
    iter_step = 20000
    '''
    Training
    '''
    for epoch in range(start_epoch+1, dict['Epoch']):
    # while True:
        log_string(f'Epoch:{epoch} strat-------')
        learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        log_string(f'lr_rate:{learning_rate}')
        num_batches = len(train_loader)
        loss_sum = 0
        total_point = 0
        model = model.train()
        print(f'cuda memorry:{torch.cuda.memory_allocated(device=opt.gpu) / (1024*1024)}')


        for i,(aff_name,img, points, labels, logits_labels, sub_box, obj_box) in enumerate(train_loader):

            optimizer.zero_grad()
            # img = img.to(device)
            B = img.size(0)
            logits_labels = logits_labels

            temp_loss = 0
            for point, label, logits_label in zip(points, labels, logits_labels):
                point, label = point.float(), label.float()
                affordance_gt = label.unsqueeze(dim=-1)
                
                if(opt.use_gpu):
                    img = img.to(device)
                    point = point.to(device)
                    affordance_gt = affordance_gt.to(device)
                    logits_label = logits_label.to(device)
                    sub_box = sub_box.to(device)
                    obj_box = obj_box.to(device)
       

                pre_affordance = model(aff_name,img, point, sub_box, obj_box)
 
                loss_a = loss_ca(pre_affordance, affordance_gt)
                temp_loss += dict['w2']*loss_a

            temp_loss.backward()
            loss_sum += temp_loss.item()
            optimizer.step()
            # scheduler.step()
            avg_loss = loss_sum / 2
            # log_string(f'total_iter: {total_iter+1} : loss: {avg_loss}')
            # if((total_iter+1) % show_step==0):
            # if (total_iter) % show_step == 0:
            #     print(f'Iteration { total_iter}/{iter_step} | loss: {temp_loss.item()}')
            #     writer.add_scalar('train_loss', temp_loss.item(), total_iter*iter_step + i)

            total_iter += 1

            print(f'Epoch:{epoch} | iteration:{i} | loss:{temp_loss.item()}')
            
            
        # if(opt.storage == True):
        #     if((epoch+1) % 1==0):
        #         model_path = save_path + '/Epoch_' + str(total_iter+1) + '.pt'
        #         checkpoint = {
        #                 'model': model.state_dict(),
        #                 'optimizer': optimizer.state_dict(),
        #                 'Epoch': total_iter+1
        #             }
        #         torch.save(checkpoint, model_path)
        #         log_string(f'model saved at {model_path}')

        if(( epoch+1)%1 == 0):
            model = model.eval()
            loss_sum = 0
            pr_aff, gt_aff = [], []

            aff_preds = torch.zeros((len(val_dataset), 2048, 1))
            aff_targets = torch.zeros((len(val_dataset), 2048, 1))

            with torch.no_grad():
                log_string(f'EVALUATION strat-------')
                for i,(aff_name,img, point, label,_,_,sub_box, obj_box) in enumerate(val_loader):
                    point, label = point.float(), label.float()
                    affordance_gt = label.unsqueeze(dim=-1)
                    if(opt.use_gpu):
                        img = img.to(device)
                        point = point.to(device)
                        affordance_gt = affordance_gt.to(device)
                        sub_box = sub_box.to(device)
                        obj_box = obj_box.to(device)
      
                    pre_affordance = model(aff_name,img, point, sub_box, obj_box)

                    loss_a = loss_ca(pre_affordance, affordance_gt)
                    temp_loss = dict['w2']*loss_a
                    loss_sum += temp_loss.item()

                    pr_aff.append(pre_affordance)
                    gt_aff.append(affordance_gt)

                aff_preds, aff_targets = torch.cat(pr_aff, 0), torch.cat(gt_aff, 0)

            AUC_, IOU_, SIM_ = evaluate(aff_preds, aff_targets)
            log_string(f'AUC:{AUC_} | IOU:{IOU_} | SIM:{SIM_}')

            if(AUC_ > best_current['AUC']):
                best_current['AUC'] = AUC_
                best_current['aIOU'] = IOU_
                best_current['SIM'] = SIM_
                best_model_path = save_path + '/best.pt'
                checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'Epoch': iter
                    }
                torch.save(checkpoint, best_model_path)
                log_string(f'best model saved at {best_model_path}')
                    # print(f'best AUC{AUC_})

            # val_loss = loss_sum / batches_val

            # log_string(f'Iteration: {total_iter + 1} : val_loss: {val_loss}')

            
            # if (total_iter + 1) >= iter_step:

            #     exit()
        scheduler.step()

            
    
def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='cuda:0', help='gpu device id')
    parser.add_argument('--decay_rate', type=float, default=1e-3, help='weight decay [default: 1e-3]')
    parser.add_argument('--use_gpu', type=str, default=True, help='whether or not use gpus')
    parser.add_argument('--save_dir', type=str, default='runs/train/', help='path to save .pt model while training')
    parser.add_argument('--name', type=str, default='DAG', help='training name to classify each training process')
    parser.add_argument('--resume', type=str, default=False, help='start training from previous epoch')
    parser.add_argument('--checkpoint_path', type=str, default='runs/train/DAG/best.pt', help='checkpoint path')
    parser.add_argument('--log_name', type=str, default='train.log', help='the name of current training')
    parser.add_argument('--loss_cls', type=float, default=0.3, help='cls loss scale')
    parser.add_argument('--loss_kl', type=float, default=0.5, help='kl loss scale')
    parser.add_argument('--storage', type=bool, default=False, help='whether to storage the model during training')
    parser.add_argument('--yaml', type=str, default='config/config_seen.yaml', help='yaml path')

    opt = parser.parse_args()
    seed_torch(seed=42)
    torch.autograd.set_detect_anomaly(True)
    dict = read_yaml(opt.yaml)
    main(opt, dict)