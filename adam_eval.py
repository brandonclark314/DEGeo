import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import torch
from models import GeoCLIP, toLatLon
from config import getopt
import numpy as np
from geopy.distance import geodesic as GD
import dataloader
import pickle
import pandas as pd
from train_and_eval import toCartesian
import json
from einops import rearrange
import wandb

def distance_accuracy(targets, preds, dis=2500, opt=None):
    ground_truth = [(x[0], x[1]) for x in targets]   

    total = len(ground_truth)
    correct = 0

    for i in range(len(ground_truth)):
        #print(GD(predictions[i], ground_truth[i]).km)
        #print(f'Ground Truth: {ground_truth[i]}, Prediction: {predictions[i]}')
        if GD(preds[i], ground_truth[i]).km <= dis:
            correct += 1

    return correct / total

def toCartesianVec(L):
    L = L * np.pi / 180

    x = torch.cos(L[:, 0]) * torch.cos(L[:, 1])
    y = torch.cos(L[:, 0]) * torch.sin(L[:, 1])
    z = torch.sin(L[:, 0])
    
    R = torch.stack([x, y, z], dim=1)
    return R

def toLatLonVec(R):
    x = R[:, 0]
    y = R[:, 1]
    z = R[:, 2]
    
    lat = torch.arctan2(z, torch.sqrt(x**2 + y**2))
    lon = torch.arctan2(y, x)
    
    lat = lat * 180 / np.pi
    lon = lon * 180 / np.pi
    
    L = torch.stack([lat, lon], dim=1)
    return L

def adam_eval(val_dataloader, model, epoch, opt):
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
    
    for parameters in model.parameters():
        parameters.requires_grad = False
    
    for i, (imgs, labels, scenes) in bar:
        labels = labels.cpu().numpy()
        imgs = imgs.to(opt.device)
        
        locations = toLatLonVec(locations)
        
        # Define Optimization Config
        locations.requires_grad = True
        optimizer = torch.optim.Adam([locations], lr=0.001)
        loss = 0
        
        for j in range(opt.eval_steps):
            optimizer.zero_grad()
    
            # Get predictions (probabilities for each location based on similarity)
            logits_per_image, logits_per_location, scene_pred, \
                img_momentum_matrix, gps_momentum_matrix = model(imgs, toCartesianVec(locations))
            probs = logits_per_image.softmax(dim=-1)
    
            loss = -torch.log(probs).mean()
            loss.backward()
            optimizer.step()
            
        # Predict gps location with the highest probability (index)
        outs = torch.argmax(probs, dim=-1).detach().cpu().numpy()
        
            
        # Save the predictions and targets
        targets.append(labels)
        preds.append(locations[outs])
        

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    model.train()
    for parameters in model.parameters():
        parameters.requires_grad = True

    accuracies = []
    for dis in opt.distances:
        acc = distance_accuracy(targets, preds, dis=dis, opt=opt)
        print("Accuracy", dis, "is", acc)
        wandb.log({opt.testset + " " +  str(dis) + " Accuracy" : acc})

if __name__ == '__main__':

    opt = getopt()

    #opt.device = torch.device('cpu')

    model = GeoCLIP(opt=opt)
    model.load_state_dict(torch.load(opt.saved_model), strict=False, map_location=opt.device)
    _ = model.to(opt.device)

    val_dataset = dataloader.M16Dataset(split=opt.testset, opt=opt)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=opt.kernels, shuffle=True, drop_last=False)
