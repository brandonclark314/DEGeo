import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import torch
from models import GeoCLIP
from config import getopt
import numpy as np
from geopy.distance import geodesic as GD
from coordinates import toCartesian, toLatLon
import wandb
import dataloader
import pickle
from global_land_mask import globe

def to_lat_lon(point=None):
    lat = np.arcsin(point[2])*180/np.pi
    lon = np.arcsin(point[1] / np.cos(np.arcsin(point[2])))*180/np.pi
    return lat.item(), lon.item()

def distance(pt1, pt2):
    """Given Torch Tensors pt1 and pt2, return the R2 distance between them."""
    return torch.norm(pt1 - pt2)

def filterLand(points):
    print("Filtering land", flush=True)
    points_lat_lon = toLatLon(points)
    lat = points_lat_lon[:, 0]
    lon = points_lat_lon[:, 1]
    
    lat = lat.cpu().numpy()
    lon = lon.cpu().numpy()
    
    isLand = globe.is_land(lat, lon)
    
    points = points[isLand]
    
    print("Original Points:", points_lat_lon.shape[0], flush=True)
    print("Filtered Points:", points.shape[0], flush=True)
    return points

def fibonacci_sphere(samples=1000, preds=None, dists=None, opt=None):
    if dists:
        points = [[] for i in range(opt.batch_size)]
    else:
        points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in tqdm(range(samples)):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        if dists:
            distances = torch.cdist(preds,torch.Tensor([x,y,z]).reshape(1,1,3).to(opt.device), p=2)
            for j in range(opt.batch_size):
                for k in range(opt.num_samples):
                    if distances[j][k] < dists[j][k]:
                        if (x,y,z) not in points[j]:
                            points[j].append((x,y,z))
        else:
            points.append((x, y, z))

    return points

def distance_accuracy(targets, preds, dis=2500, opt=None):
    ground_truth = [(x[0], x[1]) for x in targets]   
    preds = [(x[0], x[1]) for x in preds]

    total = len(ground_truth)
    correct = 0

    for i in range(len(ground_truth)):
        if GD(preds[i], ground_truth[i]).km <= dis:
            correct += 1

    return correct / total

def fibonacci_eval(val_dataloader, model, epoch, opt):
    bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
    
    # Save all the classes (possible locations to predict)
    locations = fibonacci_sphere(samples=1000, dists=None, opt=opt)
    locations = torch.tensor(locations)
    locations = locations.to(opt.device)
    locations = filterLand(locations)

    preds = []
    targets = []

    model.eval()
    
    for i, (imgs, labels, scenes) in bar:
        labels = labels.cpu().numpy()
        imgs = imgs.to(opt.device)
        
        # First Prediction
        logits_per_image, logits_per_location, scene_pred, \
            img_momentum_matrix, gps_momentum_matrix = model(imgs, locations)
        probs = logits_per_image.softmax(dim=-1)
        
        # Last Prediction
        logits_per_image, logits_per_location, scene_pred, \
            img_momentum_matrix, gps_momentum_matrix = model(imgs, locations)
        probs = logits_per_image.softmax(dim=-1)

        outs = torch.argmax(probs, dim=-1).detach().cpu().numpy()
        
        # Save the predictions and targets
        targets.append(labels)
        preds.append(toLatLon(locations[outs]).detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    model.train()

    accuracies = []
    for dis in opt.distances:
        acc = distance_accuracy(targets, preds, dis=dis, opt=opt)
        print("Accuracy", dis, "is", acc)
        wandb.log({opt.testset + " " +  str(dis) + " Accuracy" : acc})
