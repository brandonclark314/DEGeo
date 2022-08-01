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

def loc_enc_eval(val_dataloader, model, epoch, opt):
    bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
    
    # Save all the classes (possible locations to predict)
    fine_gps = pd.read_csv(opt.resources + "cells_50_1000_images_4249548.csv")
    locations = list(fine_gps.loc[:, ['latitude_mean', 'longitude_mean']].to_records(index=False))
    locations = [toCartesian(x[0], x[1]) for x in locations]
    locations = torch.tensor(locations)
    locations = locations.to(opt.device)

    preds_1km = []
    preds_25km = []
    preds_200km = []
    preds_750km = []
    preds_2500km = []
    
    targets_1km = []
    targets_25km = []
    targets_200km = []
    targets_750km = []
    targets_2500km = []

    model.eval()
    
    for i, (imgs, labels, scenes) in bar:
        labels = labels.cpu().numpy()
        imgs = imgs.to(opt.device)
        
        # Compute Features
        image_features = model.image_encoder(imgs)
        
        locations = locations.float()
        
        # Location Features: 1km, 25km, 200km, 750km, 2500km
        location_features_1km = model.location_encoder.LocEnc1k(locations)
        location_features_25km = model.location_encoder.LocEnc25k(locations)
        location_features_200km = model.location_encoder.LocEnc200k(locations)
        location_features_750km = model.location_encoder.LocEnc750k(locations)
        location_features_2500km = model.location_encoder.LocEnc2500k(locations)
        
        # Normalize Features
        image_features = F.normalize(image_features, dim=1)
        location_features_1km = F.normalize(location_features_1km, dim=1)
        location_features_25km = F.normalize(location_features_25km, dim=1)
        location_features_200km = F.normalize(location_features_200km, dim=1)
        location_features_750km = F.normalize(location_features_750km, dim=1)
        location_features_2500km = F.normalize(location_features_2500km, dim=1)
        
        # Compute Similarities
        logit_scale = model.logit_scale.exp()
        
        logits_per_image_1km = logit_scale * (image_features @ location_features_1km.t())
        logits_per_image_25km = logit_scale * (image_features @ location_features_25km.t())
        logits_per_image_200km = logit_scale * (image_features @ location_features_200km.t())
        logits_per_image_750km = logit_scale * (image_features @ location_features_750km.t())
        logits_per_image_2500km = logit_scale * (image_features @ location_features_2500km.t())
            
        probs_1km = logits_per_image_1km.softmax(dim=-1)
        probs_25km = logits_per_image_25km.softmax(dim=-1)
        probs_200km = logits_per_image_200km.softmax(dim=-1)
        probs_750km = logits_per_image_750km.softmax(dim=-1)
        probs_2500km = logits_per_image_2500km.softmax(dim=-1)
        
        outs_1km = torch.argmax(probs_1km, dim=-1).detach().cpu().numpy()
        outs_25km = torch.argmax(probs_25km, dim=-1).detach().cpu().numpy()
        outs_200km = torch.argmax(probs_200km, dim=-1).detach().cpu().numpy()
        outs_750km = torch.argmax(probs_750km, dim=-1).detach().cpu().numpy()
        outs_2500km = torch.argmax(probs_2500km, dim=-1).detach().cpu().numpy()
        
        # GPS Coordinates
        gps_1km = locations[outs_1km]
        gps_25km = locations[outs_25km]
        gps_200km = locations[outs_200km]
        gps_750km = locations[outs_750km]
        gps_2500km = locations[outs_2500km]
        
        # Save the predictions and targets
        targets_1km.append(outs_1km)
        targets_25km.append(outs_25km)
        targets_200km.append(outs_200km)
        targets_750km.append(outs_750km)
        targets_2500km.append(outs_2500km)
        
        preds_1km.append(gps_1km.detach().cpu().numpy())
        preds_25km.append(gps_25km.detach().cpu().numpy())
        preds_200km.append(gps_200km.detach().cpu().numpy())
        preds_750km.append(gps_750km.detach().cpu().numpy())
        preds_2500km.append(gps_2500km.detach().cpu().numpy())
        

    preds_1km = np.concatenate(preds_1km, axis=0)
    preds_25km = np.concatenate(preds_25km, axis=0)
    preds_200km = np.concatenate(preds_200km, axis=0)
    preds_750km = np.concatenate(preds_750km, axis=0)
    preds_2500km = np.concatenate(preds_2500km, axis=0)
    
    targets_1km = np.concatenate(targets_1km, axis=0)
    targets_25km = np.concatenate(targets_25km, axis=0)
    targets_200km = np.concatenate(targets_200km, axis=0)
    targets_750km = np.concatenate(targets_750km, axis=0)
    targets_2500km = np.concatenate(targets_2500km, axis=0)
    
    preds = [preds_1km, preds_25km, preds_200km, preds_750km, preds_2500km]
    targets = [targets_1km, targets_25km, targets_200km, targets_750km, targets_2500km]
    
    model.train()

    accuracies = []
    for pred, target, name in zip(preds, targets, ['1km', '25km', '200km', '750km', '2500km']):
        for dis in opt.distances:
            acc = distance_accuracy(target, pred, dis=dis, opt=opt)
            print(f'LocEnc{name} Accuracy', dis, "is", acc)
            wandb.log({opt.testset + " " +  str(dis) + f' LocEnc{name} Accuracy' : acc})

if __name__ == '__main__':

    opt = getopt()

    #opt.device = torch.device('cpu')

    model = GeoCLIP(opt=opt)
    model.load_state_dict(torch.load(opt.saved_model), strict=False, map_location=opt.device)
    _ = model.to(opt.device)

    val_dataset = dataloader.M16Dataset(split=opt.testset, opt=opt)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=opt.kernels, shuffle=True, drop_last=False)
