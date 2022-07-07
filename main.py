import os, numpy as np, argparse, time, multiprocessing
from tqdm import tqdm

import torch
import torch.nn as nn

import dataloader
from train_and_eval import train_images, eval_images

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
        entity='brandonclark314',
        config=config)
wandb.run.name = opt.description
wandb.save()

train_dataset = dataloader.M16Dataset(split=opt.trainset, opt=opt)
val_dataset = dataloader.M16Dataset(split=opt.testset, opt=opt)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.kernels, shuffle=True, drop_last=False)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=opt.kernels, shuffle=True, drop_last=False)

img_criterion = torch.nn.CrossEntropyLoss()
scene_criterion = torch.nn.CrossEntropyLoss()

model = models.GeoCLIP(opt=opt)

optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=0.5)

_ = model.to(opt.device)
wandb.watch(model, img_criterion, log="all")
wandb.watch(model, gps_criterion, log="all")

if not os.path.exists('./weights/'):
    os.mkdir('./weights/')

best_loss = 10000
for epoch in range(opt.n_epochs): 
    eval_images(val_dataloader=val_dataloader, model=model, epoch=epoch, opt=opt)

    if not opt.evaluate:
        _ = model.train()

        loss = train_images(train_dataloader=train_dataloader, model=model, img_criterion=img_criterion, gps_criterion=gps_criterion, optimizer=optimizer, scheduler=scheduler, opt=opt, epoch=epoch, val_dataloader=val_dataloader)

    torch.save(model.state_dict(), 'weights/' + opt.description + '.pth')

    if loss < best_loss:
        best_loss = loss
        loss = round(loss, 2)
        torch.save(model.state_dict(), 'weights/' + opt.description + '_' + str(epoch) + '_' + str(loss) + '.pth')

    scheduler.step()
