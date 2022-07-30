import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import torch
from models import GeoCLIP
from config import getopt
import numpy as np
from geopy.distance import geodesic as GD
import dataloader
import pickle
import pandas as pd
from train_and_eval import toCartesian
import json
from einops import rearrange

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

def to_lat_lon(point=None):
    lat = np.arcsin(point[2])*180/np.pi
    lon = np.arcsin(point[1] / np.cos(np.arcsin(point[2])))*180/np.pi
    return lat.item(), lon.item()

def cell_zoom(val_dataloader, model, epoch, opt):
    bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
    
    min_max = json.load(open('class_min_max.json','r'))
    # Save all the classes (possible locations to predict)
    fine_gps = pd.read_csv(opt.resources + "cells_50_1000_images_4249548.csv")
    locations = list(fine_gps.loc[:, ['latitude_mean', 'longitude_mean']].to_records(index=False))
    locations = [toCartesian(x[0], x[1]) for x in locations]
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
            logits_per_image, logits_per_location, scene_pred, \
                img_momentum_matrix, gps_momentum_matrix = model(imgs, locations)
            probs = logits_per_image.softmax(dim=-1)
        
        
        # Predict gps location with the highest probability (index)
        outs = torch.argmax(probs, dim=-1).detach().cpu().numpy()

        idx = 0
        class_to_indices = {}
        img_to_indices = {}
        new_locations = []

        for j in range(imgs.shape[0]):
            if outs[j] in class_to_indices:
                img_to_indices[j] = class_to_indices[outs[j]]
            else:
                class_to_indices[outs[j]] = (0 + 1024*idx, 1024*(idx+1))
                img_to_indices[j] = (0 + 1024*idx, 1024*(idx+1))
                idx += 1
                lat_min = min_max[str(outs[j])]['lat_min']
                lat_max = min_max[str(outs[j])]['lat_max']
                lon_min = min_max[str(outs[j])]['lon_min']
                lon_max = min_max[str(outs[j])]['lon_max']

                del_x = (lat_max-lat_min) / 32
                del_y = (lon_max-lon_min) / 32

                for k in range(32):
                    lat = lat_min + del_x * k
                    for l in range(32):
                        lon = lon_min + del_y * l
                        new_locations.append([lat, lon])
        
        new_locations_cart = [toCartesian(x[0], x[1]) for x in new_locations]
        new_locations_cart = torch.tensor(new_locations_cart)
        new_locations_cart = new_locations_cart.to(opt.device)

        with torch.no_grad():
            logits_per_image, logits_per_location, scene_pred = model(imgs, new_locations_cart)
        
            for j in range(imgs.shape[0]):
                select = rearrange(logits_per_image[j,img_to_indices[j][0]:img_to_indices[j][1]], '(bs dim) -> bs dim', bs=1)
                new_outs = select.softmax(dim=-1)
                final_outs = torch.argmax(new_outs, dim=-1).detach().cpu().numpy()
                start = class_to_indices[outs[j]][0]

                preds.append(new_locations[final_outs[0]+start])
            
            targets.append(labels)


    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    model.train()

    accuracies = []
    for dis in opt.distances:
        acc = distance_accuracy(targets, preds, dis=dis, opt=opt)
        print("Accuracy", dis, "is", acc)
        #wandb.log({opt.testset + " " +  str(dis) + " Accuracy" : acc})

if __name__ == '__main__':

    opt = getopt()

    #opt.device = torch.device('cpu')

    model = GeoCLIP(opt=opt)
    model.load_state_dict(torch.load(opt.saved_model), strict=False, map_location=opt.device)
    _ = model.to(opt.device)

    val_dataset = dataloader.M16Dataset(split=opt.testset, opt=opt)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=opt.kernels, shuffle=True, drop_last=False)

    cell_zoom(val_dataloader, model, 0, opt)
