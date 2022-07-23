from cmath import exp
import time
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from colorama import Fore, Style
from infonce import InfoNCE

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

discretize = np.vectorize(lambda x, alpha: 1 if x > alpha else -1)

# Numpy version of the function
def toCartesian(latitude, longitude):
    lat = latitude * np.pi / 180
    lon = longitude * np.pi / 180
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return x, y, z

toCartesianVec = np.vectorize(toCartesian)

def toLatLon(x, y, z):
    # Unit sphere to GPS
    lat = np.arctan2(z, np.sqrt(x**2 + y**2))
    lon = np.arctan2(y, x)
    
    # Go to degrees
    lat = lat * 180 / np.pi
    lon = lon * 180 / np.pi
    
    return [lat, lon]

def getRandomCoordinates(num_coords):
    coords = 2 * torch.rand(num_coords, 3) - 1
    coords = coords / coords.norm(dim=1, keepdim=True)
    return coords

def log_sim_loss(y_true, y_pred, opt):
    earth_radius = 6371
    y_true = y_true.float()
    y_pred = y_pred.float()

    cos_sim = cos_sim = torch.nn.CosineSimilarity()(y_true, y_pred)
    km = torch.acos(cos_sim) * earth_radius
    cos_sim = torch.nn.CosineSimilarity()(y_true, y_pred)

    km = torch.acos(torch.mean(cos_sim)) * earth_radius

    cos_sim_squeezed = (cos_sim + 1) / 2
    log_sim_loss = torch.mean(-torch.log(cos_sim_squeezed)).to(opt.device)

    return log_sim_loss, km

def train_images(train_dataloader, model, img_criterion, scene_criterion, optimizer, scheduler, opt, epoch, val_dataloader=None):

    batch_times, model_times, losses = [], [], []
    accuracy_regressor, accuracy_classifier = [], []
    tt_batch = time.time()

    data_iterator = train_dataloader 

    losses = []
    running_loss = 0.0
    dataset_size = 0
    gps_multiplier = model.GPS_Aug_Multiplier


    val_cycle = (len(data_iterator.dataset.data) // (opt.batch_size * 25))
    print("Outputting loss every", val_cycle, "batches")
    print("Validating every", val_cycle, "batches")
    print("Starting Epoch", epoch)

    bar = tqdm(enumerate(data_iterator), total=len(data_iterator))

    for i ,(imgs, classes, scenes) in bar:
        batch_size = imgs.shape[0]

        if opt.traintype == 'CLIP':
            gps = classes.to(opt.device)
        if opt.traintype == 'Classification':
            coarse = classes[:,0].type(torch.LongTensor)
            coarse = coarse.to(opt.device)
            medium = classes[:,1].type(torch.LongTensor)
            medium = medium.to(opt.device)
            fine = classes[:,2].type(torch.LongTensor)
            fine = fine.to(opt.device)
        
        imgs = imgs.to(opt.device)
        
        # Add extra GPS Coordinates
        extra_gps = getRandomCoordinates(batch_size * gps_multiplier).to(opt.device)
        gps_aug = torch.cat((gps, extra_gps), dim=0)
        
        scene_labels3 = scenes[:, 0]
        scene_labels16 = scenes[:, 1]
        scene_labels365 = scenes[:, 2]
        
        scene_labels3 = scene_labels3.to(opt.device)
        scene_labels16 = scene_labels16.to(opt.device)
        scene_labels365 = scene_labels365.to(opt.device)

        optimizer.zero_grad()
        
        if opt.traintype == 'CLIP':
            img_matrix, gps_matrix, scene_pred, gps_0 = model(imgs, gps_aug)
            targets = torch.cat((torch.eye(batch_size), torch.zeros(batch_size,
                                                                    batch_size * gps_multiplier)), dim=1).to(opt.device)
        if opt.traintype == 'Classification':
            out1, out2, out3 = model(imgs)

        torch.set_printoptions(edgeitems=30)
    
        # Compute the loss
        loss = 0
        if opt.traintype == 'CLIP':
            img_loss = img_criterion(img_matrix, targets).float()
            gps_loss = img_criterion(gps_matrix.t(), targets).float()
            gps_pred_loss, km = log_sim_loss(gps, gps_0, opt=opt)
        
            if opt.scene:
                scene_loss = (scene_criterion(scene_pred[0], scene_labels3).float() +
                              scene_criterion(scene_pred[1], scene_labels16).float() +
                              scene_criterion(scene_pred[2], scene_labels365).float()) / 3
                
                loss = (img_loss + gps_loss + scene_loss) / 3
            else:
                loss = (img_loss + gps_loss + gps_pred_loss) / 3
        if opt.traintype == 'Classification':
            loss1 = img_criterion(out1, coarse)
            loss2 = img_criterion(out2, medium)
            loss3 = img_criterion(out3, fine)

            loss = loss1 + loss2 + loss3

            if opt.scene:
                loss += scene_criterion(scene_pred)
                loss = loss / 4

        loss.backward()

        optimizer.step()     
        # scheduler.step()

        losses.append(loss.item())
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
        
        if i % val_cycle == 0:
            if opt.traintype == 'CLIP':
                wandb.log({"Training Loss" : loss.item()})
                wandb.log({"Image Loss": img_loss.item()})
                wandb.log({"GPS Pred. Arc": km.item()})
            if opt.traintype == 'Classification':
                wandb.log({"Classification Loss" : loss.item()})
            if opt.scene:
                wandb.log({"Scene Loss": scene_loss.item()})
            #print("interation", i, "of", len(data_iterator))
        if False and val_dataloader != None and i % val_cycle == 0:
            if opt.hier_eval:
                eval_images_weighted(val_dataloader, model, epoch, opt)
            else:
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

    if opt.partition == 'fine':
        predictions = list(fine_gps.iloc[preds][['latitude_mean', 'longitude_mean']].to_records(index=False))
    elif opt.partition == '3K':
        predictions = dataloader.get_im2gps3k_test_classes(opt=opt, cartesian_coords=False) 
        predictions = [predictions[i] for i in preds]
    elif opt.partition == 'Mix':
        locations = list(fine_gps.loc[:, ['latitude_mean', 'longitude_mean']].to_records(index=False))
        locations += dataloader.get_im2gps3k_test_classes(opt=opt, cartesian_coords=False)
        predictions = [locations[i] for i in preds]
    
    ground_truth = [(x[0], x[1]) for x in targets]   

    total = len(ground_truth)
    correct = 0

    for i in range(len(ground_truth)):
        #print(GD(predictions[i], ground_truth[i]).km)
        #print(f'Ground Truth: {ground_truth[i]}, Prediction: {predictions[i]}')
        if GD(predictions[i], ground_truth[i]).km <= dis:
            correct += 1

    return correct / total

def eval_images(val_dataloader, model, epoch, opt):
    bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
    
     # Save all the classes (possible locations to predict)
     
    if opt.partition == 'fine':
        fine_gps = pd.read_csv(opt.resources + "cells_50_1000_images_4249548.csv")
        locations = list(fine_gps.loc[:, ['latitude_mean', 'longitude_mean']].to_records(index=False))
        locations = [toCartesian(x[0], x[1]) for x in locations]
    elif opt.partition == '3K':
        locations = dataloader.get_im2gps3k_test_classes(opt=opt, cartesian_coords=True)
    elif opt.partition == 'Mix':
        fine_gps = pd.read_csv(opt.resources + "cells_50_1000_images_4249548.csv")
        locations = list(fine_gps.loc[:, ['latitude_mean', 'longitude_mean']].to_records(index=False))
        locations = [toCartesian(x[0], x[1]) for x in locations]
        locations += dataloader.get_im2gps3k_test_classes(opt=opt, cartesian_coords=True)
    
    locations = torch.tensor(locations)
    locations = locations.to(opt.device)

    preds = []
    targets = []

    model.eval()
    
    for i, (imgs, labels, scenes) in bar:
        labels = labels.cpu().numpy()
        imgs = imgs.to(opt.device)
        
        # Get predictions (probabilities for each location based on similarity)
        with torch.no_grad():
            if opt.traintype == 'CLIP':
                logits_per_image, logits_per_location, scene_pred = model(imgs, locations)
            if opt.traintype == 'Classification':
                logits_per_image = model(imgs)
        probs = logits_per_image.softmax(dim=-1)
        
        # Predict gps location with the highest probability (index)
        outs = torch.argmax(probs, dim=-1).detach().cpu().numpy()
        
        # Save the predictions and targets
        targets.append(labels)
        preds.append(outs)

    print("Shape Locations 1:", locations.shape)
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    model.train()

    accuracies = []
    for dis in opt.distances:
        acc = distance_accuracy(targets, preds, dis=dis, opt=opt)
        print("Accuracy", dis, "is", acc)
        wandb.log({opt.testset + " " +  str(dis) + " Accuracy" : acc})
        
def distance_accuracy_direct(targets, preds, dis=2500, set='im2gps3k', trainset='train', opt=None):
    ground_truth = [(x[0], x[1]) for x in targets]   

    total = len(ground_truth)
    correct = 0

    for i in range(len(ground_truth)):

        if GD(preds[i], ground_truth[i]).km <= dis:
            correct += 1

    return correct / total
        
def eval_images_SGD(val_dataloader, model, epoch, opt):
    bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))

    preds = []
    targets = []

    model.eval()
    
    for i, (imgs, labels, scenes) in bar:
        labels = labels.cpu().numpy()
        imgs = imgs.to(opt.device)
        
        outs = model.predict(imgs).detach().cpu().numpy()
        
        # Save the predictions and targets
        targets.append(labels)
        preds.append(outs)

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    model.train()

    accuracies = []
    for dis in opt.distances:
        acc = distance_accuracy_direct(targets, preds, dis=dis, opt=opt)
        print("Accuracy", dis, "is", acc)
        wandb.log({opt.testset + " " +  str(dis) + " Accuracy" : acc})

def eval_images_weighted(val_dataloader, model, epoch, opt):
    data_iterator = val_dataloader

    bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))

    preds = []
    targets = []

    for i, (imgs, classes) in bar:

        labels = classes.cpu().numpy()

        imgs = imgs.to(opt.device)
        with torch.no_grad():
            outs1, outs2, outs3 = model(imgs, evaluate=True)

        outs1 = F.softmax(outs1, dim=1)
        outs2= F.softmax(outs2, dim=1)
        outs3 = F.softmax(outs3, dim=1)

        coarseweights = torch.ones(outs2.shape).cuda()
        mediumweights = torch.ones(outs3.shape).cuda()

        for i in range(outs2.shape[1]):
            coarseweights[:,i] = outs1[:,val_dataloader.dataset.coarse2medium[i]]

        outs2 = outs2 * coarseweights

        for i in range(outs3.shape[1]):
            mediumweights[:,i] = outs2[:,val_dataloader.dataset.medium2fine[i]]
        outs3 = outs3 * mediumweights

        outs3 = torch.argmax(outs3, dim=-1).detach().cpu().numpy()

        targets.append(labels)
        preds.append(outs3)

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    '''
    macrof1 = f1_score(targets, preds, average='macro')
    weightedf1 = f1_score(targets, preds, average='weighted')
    accuracy =  accuracy_score(targets, preds)
    '''
    #np.set_printoptions(precision=15)
    #print(targets)
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

    # Setting this low for testing
    opt.batch_size = 4

    # Load Model
    model = models.GeoCLIP()

    # Create val dataloader for testing
    val_dataset = dataloader.M16Dataset(split=opt.testset, opt=opt)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=opt.kernels, shuffle=False, drop_last=False)
    
    # Run evaluation code
    eval_images(val_dataloader, model, 0, opt)
    exit()

    # Generate Random Data
    locations = torch.randn((1000, 3)) # Possible Locations
    labels = torch.rand((10, 3)) # Latitude and Longitude
    imgs = torch.rand((10, 3, 224, 224)) # Images
    
    # Move to Device
    labels = labels.cpu().numpy()
    imgs = imgs.to(opt.device)

    accuracies = []
    for dis in opt.distances:
        acc = distance_accuracy(targets, preds, dis=dis, opt=opt)
        print("Accuracy", dis, "is", acc)
        wandb.log({opt.testset + " " +  str(dis) + " Accuracy" : acc})
    
    
   
    
    
    
    
