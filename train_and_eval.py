import time
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from colorama import Fore, Style

from einops import rearrange

import torch
import torch.nn.functional as F
import pickle

import geopy
from geopy.distance import geodesic as GD
from tqdm import tqdm

import wandb
#import evaluate
import pandas as pd
import json

import models
from config import getopt
import dataloader


def train_images(train_dataloader, model, criterion, optimizer, scheduler, opt, epoch, val_dataloader=None):

    batch_times, model_times, losses = [], [], []
    accuracy_regressor, accuracy_classifier = [], []
    tt_batch = time.time()

    data_iterator = train_dataloader 

    losses = []
    running_loss = 0.0
    dataset_size = 0


    val_cycle = (len(data_iterator.dataset.data) // (opt.batch_size * 164))
    print("Outputting loss every", val_cycle, "batches")
    #print("Validating every", val_cycle*5, "batches")
    print("Starting Epoch", epoch)

    bar = tqdm(enumerate(data_iterator), total=len(data_iterator))

    for i ,(imgs, gps) in bar:

        batch_size = imgs.shape[0]

        # labels = torch.Tensor([x for x in range(batch_size)])
        # labels = labels.type(torch.LongTensor)
        # labels = labels.to(opt.device)

        gps = gps.to(opt.device)
        imgs = imgs.to(opt.device)

        optimizer.zero_grad()
        img_matrix, gps_matrix = model(imgs, gps)
        
        targets = F.softmax(
            (img_matrix + gps_matrix) / 2, dim=-1
        )
        targets = targets.to(opt.device)

        torch.set_printoptions(edgeitems=30)

        loss = 0
        img_loss = criterion(img_matrix, targets)
        gps_loss = criterion(gps_matrix, targets.T)

        loss = (img_loss + gps_loss) / 2

        loss.backward()

        optimizer.step()     
        #scheduler.step()

        losses.append(loss.item())

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
        
        if i % val_cycle == 0:
            wandb.log({"Training Loss" : loss.item()})
            wandb.log({"Image Loss": img_loss.item()})
            wandb.log({"GPS Loss": gps_loss.item()})
            #print("interation", i, "of", len(data_iterator))
        if val_dataloader != None and i % (val_cycle * 5) == 0:
            eval_images(val_dataloader, model, epoch, opt)
    
    print("The loss of epoch", epoch, "was ", np.mean(losses))
    return np.mean(losses)
    
def distance_accuracy(targets, preds, dis=2500, set='im2gps3k', trainset='train', opt=None):
    if trainset == 'train':
        coarse_gps = pd.read_csv(opt.resources + "cells_50_5000_images_4249548.csv")
        medium_gps = pd.read_csv(opt.resources + "cells_50_2000_images_4249548.csv")
        fine_gps = pd.read_csv(opt.resources + "cells_50_1000_images_4249548.csv")
    if trainset == 'train1M':
        coarse_gps = pd.read_csv(opt.resources + "cells_50_5000_images_1M.csv")

    course_preds = list(fine_gps.iloc[preds][['latitude_mean', 'longitude_mean']].to_records(index=False))
    course_target = [(x[0], x[1]) for x in targets]   

    total = len(course_target)
    correct = 0

    for i in range(len(course_target)):
        #print(GD(course_preds[i], course_target[i]).km)
        if GD(course_preds[i], course_target[i]).km <= dis:
            correct += 1

    return correct / total

def eval_images(val_dataloader, model, epoch, opt):

    data_iterator = val_dataloader

    bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))

    preds = []
    targets = []

    for i, (imgs, classes) in bar:

        labels = classes.cpu().numpy()

        imgs = imgs.to(opt.device)
        with torch.no_grad():
            outs1, outs2, outs3 = model(imgs)
        outs = torch.argmax(outs3, dim=-1).detach().cpu().numpy()

        targets.append(labels)
        preds.append(outs)

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    accuracies = []
    for dis in opt.distances:

        acc = distance_accuracy(targets, preds, dis=dis, trainset=opt.trainset, opt=opt)
        print("Accuracy", dis, "is", acc)
        wandb.log({opt.testset + " " +  str(dis) + " Accuracy" : acc})

if __name__ == '__main__':
    opt = getopt()

    opt.device = torch.device('cpu')

    model = networks.GeoGuess1(trainset=opt.trainset)
    model_pairs = model.state_dict()

    pt = torch.load('/home/c3-0/al209167/GeoGuessNet/weights/GeoGuessNet1-4.2M-Im2GPS3k-F*.pth', map_location=opt.device)
    
    for name, weight in pt.items():
        model_pairs[name] = weight

    val_dataset = dataloader.M16Dataset(split=opt.testset, opt=opt)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=opt.kernels, shuffle=False, drop_last=False)

    hierarchical_eval(val_dataloader, model, 0, opt)
