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
import dataloader
import pickle

def to_lat_lon(point=None):
    lat = np.arcsin(point[2])*180/np.pi
    lon = np.arcsin(point[1] / np.cos(np.arcsin(point[2])))*180/np.pi
    return lat.item(), lon.item()

def distance(pt1, pt2):
    """Given Torch Tensors pt1 and pt2, return the R2 distance between them."""
    return torch.norm(pt1 - pt2)

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

def closest(ref_pt,all_pts=None):
    top = {"point":None, "dist":999}

    for pt in all_pts:
        dist = distance(ref_pt, pt)
        if dist != 0:
            if dist < top["dist"]:
                top["point"] = pt
                top["dist"] = dist
    
    return top

def zoom_eval(val_dataloader, model, epoch, opt):
    bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))

    points = torch.Tensor(fibonacci_sphere(samples=1000, opt=opt)).to(opt.device)
    lengths = None

    all_preds = []
    targets = []

    model.eval()
    
    for i, (imgs, labels, scenes) in bar:
        labels = labels.cpu().numpy()
        imgs = imgs.to(opt.device)

        with torch.no_grad():
            for r in range(1, 3):
                print('Model Forward Pass', flush=True)
                logits_per_image, logits_per_location, scene = model(imgs, points)

                preds = [[] for _ in range(opt.batch_size)]

                print(f'Getting top {opt.num_samples} per image', flush=True)
                for j in range(opt.batch_size):
                    for k in range(opt.num_samples):
                        if lengths != None:
                            if j != 0:
                                tmp_preds = torch.argmax(logits_per_image[j][lengths[j-1]:lengths[j]], dim=-1)
                            else:
                                tmp_preds = torch.argmax(logits_per_image[j][:lengths[j]], dim=-1)
                        else:
                            tmp_preds = torch.argmax(logits_per_image[j], dim=-1)
                        preds[j].append(tmp_preds.item())
                        logits_per_image[j][tmp_preds.item()] = 0.0
                
                print("Grabbing Coordinates", flush=True)
                for j in range(opt.batch_size):
                    pred_coords = torch.Tensor(points[preds[j][0]]).reshape(1,3)
                    for k, idx in enumerate(preds[j]):
                        if k != 0:
                            pred_coords = torch.cat((pred_coords, points[idx].reshape(1,3)), dim=0)
                    if j == 0:
                        new_pred_coords = pred_coords.reshape(1,-1,3)
                    else:
                        new_pred_coords = torch.cat((new_pred_coords, pred_coords.reshape(1,-1,3)), dim=0)

                pred_coords = new_pred_coords.to(opt.device)

                print("Getting distance thresholds", flush=True)
                dists = torch.cdist(pred_coords, points.reshape(1,-1,3), p=2)
                smallest_dists = [[] for _ in range(opt.batch_size)]
                
                for j in tqdm(range(opt.batch_size)):
                    for k in tqdm(range(opt.num_samples)):
                        idx = torch.argmin(dists[j][k], dim=-1)
                        dists[j][k][idx] = 999
                        smallest_dists[j].append(min(dists[j][k]))

                if r == 1:
                    new_points = pickle.load(open("./weights/fib_points_100K.pkl","rb"))
                    new_points = torch.Tensor(new_points).to(opt.device)
                if r == 2:
                    new_points = pickle.load(open("./weights/fib_points_10M.pkl","rb"))
                    new_points = torch.Tensor(new_points).to(opt.device)

                print("Comparing distances to next hierarchy", flush=True)
                pts = [None for _ in range(opt.batch_size)]
                distances = torch.cdist(pred_coords, new_points, p=2)
                
                for j in range(opt.batch_size):
                    for k in range(opt.num_samples):
                        a = distances[j][k] < smallest_dists[j][k]
                        indices = a.nonzero()
                        if k == 0:
                            pts[j] = new_points[indices].reshape(-1, 3)
                        else:
                            pts[j] = torch.cat((pts[j], new_points[indices].reshape(-1, 3)), dim=0)
                
                new_points = pts

                print("Preparing new set of points", flush=True)
                lengths = [len(new_points[j]) for j in range(opt.batch_size)]

                for i in range(1, len(lengths)):
                    lengths[i] = lengths[i-1] + lengths[i]
                
                points = new_points[0]
                for j in range(1, opt.batch_size):
                    points = torch.cat((points, torch.Tensor(new_points[j])), dim=0)
                
                points = points.to(opt.device)

        logits_per_image, logits_per_location, scene = model(imgs, points)

        preds = []

        for j in range(opt.batch_size):
            if lengths != None:
                if j != 0:
                    tmp_preds = torch.argmax(logits_per_image[j][lengths[j-1]:lengths[j]], dim=-1)
                else:
                    tmp_preds = torch.argmax(logits_per_image[j][:lengths[j]], dim=-1)
            else:
                tmp_preds = torch.argmax(logits_per_image[j], dim=-1)
            preds.append(tmp_preds.item())

        pred_coords = []
        for j in range(opt.batch_size):
            pred_coords.append(to_lat_lon(points[preds[j]]))
        
        targets.append(labels)
        all_preds.append(pred_coords)
    
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    model.train()

    accuracies = []
    for dis in opt.distances:
        acc = distance_accuracy(targets, preds, dis=dis, opt=opt)
        print("Accuracy", dis, "is", acc)
        #wandb.log({opt.testset + " " +  str(dis) + " Accuracy" : acc})

def distance_accuracy(targets, preds, dis=2500, opt=None):
    ground_truth = [(x[0], x[1]) for x in targets]   

    total = len(ground_truth)
    correct = 0

    for i in range(len(ground_truth)):
        #print(GD(predictions[i], ground_truth[i]).km)
        #print(f'Ground Truth: {ground_truth[i]}, Prediction: {predictions[i]}')
        if GD(predictions[i], ground_truth[i]).km <= dis:
            correct += 1

    return correct / total

if __name__ == '__main__':

    opt = getopt()

    model = GeoCLIP(opt=opt)
    _ = model.to(opt.device)

    val_dataset = dataloader.M16Dataset(split=opt.testset, opt=opt)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=opt.kernels, shuffle=True, drop_last=False)

    zoom_eval(val_dataloader, model, 0, opt)
    exit()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D([pt[0] for pt in points[:-5]+points[-4:]], [pt[1] for pt in points[:-5]+points[-4:]], [pt[2] for pt in points[:-5]+points[-4:]], color='blue')

    ax.scatter3D([points[-5][0]], [points[-5][1]], [points[-5][2]], color='black')

    top = eight_closest(points[-5], points)

    xs = []
    ys = []
    zs = []
    for i in range(1,9):
        xs.append(top[i]["point"][0])
        ys.append(top[i]["point"][1])
        zs.append(top[i]["point"][2])



    ax.scatter3D(xs, ys, zs, color='yellow')

    new_points = fibonacci_sphere(samples=1000000, ref_pt=points[-5], dist=top[1]["dist"])

    print(len(new_points))

    random.shuffle(new_points)

    ax.scatter3D([pt[0] for pt in new_points[:100]], [pt[1] for pt in new_points[:100]], [pt[2] for pt in new_points[:100]], color='red')

    plt.savefig("test.pdf")
    #print(eight_closest(points[-1], points))
