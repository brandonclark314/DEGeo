import os, numpy as np, argparse, time, multiprocessing
from pickletools import optimize
from tqdm import tqdm

import torch
import torch.nn as nn
import pandas as pd

import dataloader
from train_and_eval import train_images, eval_images, eval_images4M, eval_images_weighted
from cell_zoom import cell_zoom
from gaussian import gaussian_eval
from loc_enc_eval import loc_enc_eval
from fibonacci_eval import fibonacci_eval
from focal_loss import FocalLoss

import wandb

import models 
from config import getopt

opt = getopt()

config = {
    'learning_rate' : opt.lr,
    'epochs' : opt.n_epochs,
    'batch_size' : opt.batch_size,
    'architecture' : opt.archname
}

wandb.init(project='DEGeo', 
        entity='vicentevivan',
        settings=wandb.Settings(start_method='fork'),
        config=config)
wandb.run.name = opt.description
wandb.save()

# MP-16
if not opt.evaluate:
    train_dataset = dataloader.M16Dataset(split=opt.trainset, opt=opt)
val_dataset = dataloader.M16Dataset(split=opt.testset, opt=opt)

if not opt.evaluate:
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.kernels, shuffle=True, drop_last=False)

val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=opt.kernels, shuffle=True, drop_last=False)

img_criterion = torch.nn.CrossEntropyLoss()
# img_criterion = FocalLoss(gamma=5.0)
scene_criterion = torch.nn.CrossEntropyLoss()

# Get Locations
def toCartesian(latitude, longitude):
    lat = latitude * np.pi / 180
    lon = longitude * np.pi / 180
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return x, y, z

# model = models.GeoCLIP(opt=opt)

GeoCLIP = models.GeoCLIP(opt=opt)
GeoCLIP.load_state_dict(torch.load(opt.saved_model))

for param in GeoCLIP.parameters():
    param.requires_grad = False

model = models.GeoCLIPLinearProbe(opt=opt, GeoCLIP=GeoCLIP)

if opt.evaluate:
    model.load_state_dict(torch.load(opt.saved_model))

optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0001)
# optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=0.0001) # Original
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=0.5)

_ = model.to(opt.device)
wandb.watch(model, img_criterion, log="all")
wandb.watch(model, scene_criterion, log="all")

if not os.path.exists('./weights/'):
    os.mkdir('./weights/')


# --------------------------------- Trainning -------------------------------- #

best_loss = 40000
for epoch in range(opt.n_epochs):
    if opt.evaluate:
        eval_images(val_dataloader=val_dataloader, model=model, epoch=epoch, opt=opt)
        break

    if opt.hier_eval:
        eval_images_weighted(val_dataloader=val_dataloader, model=model, epoch=epoch, opt=opt)
    else:
        eval_images(val_dataloader=val_dataloader, model=model, epoch=epoch, opt=opt) 

    if not opt.evaluate:
        _ = model.train()

        loss = train_images(train_dataloader=train_dataloader, model=model, img_criterion=img_criterion, scene_criterion=scene_criterion, optimizer=optimizer, scheduler=scheduler, opt=opt, epoch=epoch, val_dataloader=val_dataloader)

    torch.save(model.state_dict(), 'weights/' + opt.description + '.pth')

    if loss < best_loss:
        best_loss = loss
        loss = round(loss, 2)
        torch.save(model.state_dict(), 'weights/' + opt.description + '_' + str(epoch) + '_' + str(loss) + '.pth')

    scheduler.step()
