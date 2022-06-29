import time
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from colorama import Fore, Style

# from einops import rearrange

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
        
        targets = torch.arange(batch_size, dtype=torch.long, device=opt.device)

        torch.set_printoptions(edgeitems=30)

        loss = 0
        img_loss = criterion(img_matrix, targets)
        gps_loss = criterion(gps_matrix, targets)

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
        # coarse_gps = pd.read_csv(opt.resources + "cells_50_5000_images_4249548.csv")
        # medium_gps = pd.read_csv(opt.resources + "cells_50_2000_images_4249548.csv")
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

def toCartesian(latitude, longitude):
    lat = latitude * np.pi / 180
    lon = longitude * np.pi / 180
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return [x, y, z]

def toLatLon(x, y, z):
    lat = np.arctan2(z, np.sqrt(x**2 + y**2))
    lon = np.arctan2(y, x)
    return [lat, lon]

def eval_images(val_dataloader, model, epoch, opt):
    bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
    
     # Save all the classes (possible locations to predict)
    fine_gps = pd.read_csv(opt.resources + "cells_50_1000_images_4249548.csv")
    locations = list(fine_gps.loc[:, ['latitude_mean', 'longitude_mean']].to_records(index=False))
    locations = [toCartesian(x[0], x[1]) for x in locations]
    locations = torch.tensor(locations)
    locations = locations.to(opt.device)

    preds = []
    targets = []

    model.eval()
    
    for i, (imgs, labels) in bar:
        labels = labels.cpu().numpy()
        imgs = imgs.to(opt.device)
        
        # Get predictions (probabilities for each location based on similarity)
        with torch.no_grad():
            logits_per_image, logits_per_location = model(imgs, locations)
        
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
        # Predict gps location with the highest probability (index)
        outs = torch.argmax(probs, dim=-1).detach().cpu().numpy()
        
        # Save the predictions and targets
        targets.append(labels)
        preds.append(outs)

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    model.train()

    accuracies = []
    for dis in opt.distances:
        acc = distance_accuracy(targets, preds, dis=dis, trainset=opt.trainset, opt=opt)
        print("Accuracy", dis, "is", acc)
        wandb.log({opt.testset + " " +  str(dis) + " Accuracy" : acc})

if __name__ == '__main__':
    preds = []
    targets = []
    
    opt = getopt()

    opt.device = torch.device('cpu')

    # Load Model
    model = models.GeoCLIP()
    
    # Generate Random Data
    locations = torch.randn((1000, 3)) # Possible Locations
    labels = torch.rand((10, 3)) # Latitude and Longitude
    imgs = torch.rand((10, 3, 224, 224)) # Images
    
    # Move to Device
    labels = labels.cpu().numpy()
    imgs = imgs.to(opt.device)
    
    # # Get predictions (probabilities for each location based on similarity)
    # with torch.no_grad():
    #     model.eval()
    #     logits_per_image, logits_per_location = model(imgs, locations)
    
    # probs = logits_per_image.softmax(dim=-1)
    
    # # Predict gps location with the highest probability (index)
    # outs = torch.argmax(probs, dim=-1).detach().cpu().numpy()
    
    # print("Outputs", outs)
    
    # # Save the predictions and targets
    # targets.append(labels)
    # preds.append(outs)

    # targets = np.concatenate(targets, axis=0)
    # preds = np.concatenate(preds, axis=0)
    
    # Convert to Latitude and Longitude
    # targets = [toLatLon(x[0], x[1], x[2]) for x in targets]
    
    # for x, y in zip(targets, preds):
    #     print("Target:", x, "Prediction:", y)

    accuracies = []
    for dis in opt.distances:
        acc = distance_accuracy(targets, preds, dis=dis, opt=opt)
        print("Accuracy", dis, "is", acc)
        wandb.log({opt.testset + " " +  str(dis) + " Accuracy" : acc})
    
    
   
    
    
    
    
