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
from train_and_eval import toCartesian, toLatLon
import json
from einops import rearrange
import wandb
import torch.nn.functional as F

def distance_accuracy(targets, preds, dis=2500, opt=None):
    ground_truth = [(x[0], x[1]) for x in targets]   
    preds = [toLatLon(x[0], x[1], x[2]) for x in preds]

    total = len(ground_truth)
    correct = 0

    for i in range(len(ground_truth)):
        #print(GD(predictions[i], ground_truth[i]).km)
        #print(f'Ground Truth: {ground_truth[i]}, Prediction: {predictions[i]}')
        if GD(preds[i], ground_truth[i]).km <= dis:
            correct += 1

    return correct / total

def zoom_in(locations, n, km_std):
    """Given a set of GPS locations, sample number of locations * n from a Gaussian with mean of
    the original locations and standard deviation of std.
    """
    Earth_Diameter = 12742
    std = km_std / Earth_Diameter
    
    for i in range(n):
        print("Zoom in", i, flush=True)
        new_locations = torch.normal(locations, std=std)
        locations = torch.cat((locations, new_locations), dim=0)
        
    locations = F.normalize(locations, dim=1)
    
    return locations

def gaussian_eval(val_dataloader, model, epoch, opt):
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
        locations_opt = locations.clone().detach()
        
        labels = labels.cpu().numpy()
        imgs = imgs.to(opt.device)
        
        # First Prediction
        logits_per_image, logits_per_location, scene_pred, \
            img_momentum_matrix, gps_momentum_matrix = model(imgs, locations_opt)
        probs = logits_per_image.softmax(dim=-1)
        
        top_k, top_k_i = torch.topk(probs, k=10, dim=-1)
        locations_opt = locations_opt[torch.flatten(top_k_i)]
        
        for distance in [100, 50, 25, 10, 5, 1, 0.5, 0.25, 0.1]:
            print("\n Locations: ", locations_opt.shape, flush=True)
            locations_opt = zoom_in(locations_opt, n=15, km_std=distance)
            logits_per_image, logits_per_location, scene_pred, \
                img_momentum_matrix, gps_momentum_matrix = model(imgs, locations_opt)
            probs = logits_per_image.softmax(dim=-1)
            top_k, top_k_i = torch.topk(probs, k=10, dim=-1)
            locations_opt = locations_opt[torch.flatten(top_k_i)]
            
            print("\n Top 10: ", top_k, flush=True)
        
        # Last Prediction
        logits_per_image, logits_per_location, scene_pred, \
            img_momentum_matrix, gps_momentum_matrix = model(imgs, locations_opt)
        probs = logits_per_image.softmax(dim=-1)

        outs = torch.argmax(probs, dim=-1).detach().cpu().numpy()
        
        # Save the predictions and targets
        targets.append(labels)
        preds.append(locations_opt[outs].detach().cpu().numpy())
        

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
