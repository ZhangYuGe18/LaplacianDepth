import argparse
import os
import warnings
import numpy as np
warnings.simplefilter('ignore', np.RankWarning)
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from depth_anything.dpt import DepthAnythingV2
from util.loss import SiLogLoss
from custom_dataset import *
from train import *
root_path = "../../autodl-tmp/deepthanything"
train_image_paths, train_depth_paths,val_image_paths, val_depth_paths = get_image_path(root_path)
print(len(train_image_paths),len(train_depth_paths),len(val_image_paths),len(val_depth_paths))

import yaml
with open("config.yaml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

parser = config["parser"]
model_configs = config["model_configs"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = argparse.Namespace(**parser)

trainset = CustomDataset(image_paths=train_image_paths, depth_paths=train_depth_paths, mode='train', size=(args.img_size, args.img_size))
trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True, shuffle=True)
valset = CustomDataset(image_paths=val_image_paths, depth_paths=val_depth_paths, mode='val', size=(args.img_size, args.img_size))
valloader = DataLoader(valset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True, shuffle=True)

cudnn.enabled = True
cudnn.benchmark = True
model = DepthAnythingV2(**{**model_configs['vitb'], 'max_depth': args.max_depth})
model.load_state_dict(torch.load(args.pretrained_from, map_location=device),strict=False)

model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model.to(device)
criterion = SiLogLoss().to(device)
optimizer = AdamW([{'params': [param for name, param in model.named_parameters() if 'pretrained' in name], 'lr': args.lr},
                   {'params': [param for name, param in model.named_parameters() if 'pretrained' not in name], 'lr': args.lr * 10.0}],
                  lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100, 'log10': 100, 'silog': 100}

running_loss = train(model,args,trainloader,valloader,optimizer,criterion,device)