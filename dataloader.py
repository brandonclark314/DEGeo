import csv

from geopy.geocoders import Nominatim

from torch.utils.data import Dataset

from PIL import Image as im
import os
import torch

import pandas as pd

import numpy as np
import glob
import random

import torchvision.transforms as transforms 
from torchvision.utils import save_image

import pickle

# from einops import rearrange
import csv

import json
from collections import Counter

import matplotlib.pyplot as plt
from os.path import exists

from config import getopt
from tqdm import tqdm

def toCartesian(latitude, longitude):
    lat = latitude * np.pi / 180
    lon = longitude * np.pi / 180
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return [x, y, z]

def m16_transform():
    m16_transform_list = transforms.Compose([
        transforms.RandomAffine((1, 15)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return m16_transform_list

def m16_val_transform():
    m16_transform_list = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    return m16_transform_list    

def get_mp16_train(classfile="mp16_places365.csv", opt=None, cartesian_coords=True):

    class_info = pd.read_csv(opt.resources + classfile)
    data = json.load(open(opt.resources + 'mp16_places365_mapping_h3.json','r'))

    base_folder = opt.mp16folder

    fnames = []
    classes = []
    scenes = []

    for row in tqdm(class_info.iterrows()):
        filename = base_folder + row[1]['IMG_ID']
        if exists(filename):
            fnames.append(filename)
            
            latitude = float(row[1]['LAT'])
            longitude = float(row[1]['LON'])
            
            scenes.append([row[1]['S3_Label'], row[1]['S16_Label'], row[1]['S365_Label']])
             
            if opt.traintype ==  'CLIP':
                if cartesian_coords:
                    classes.append(toCartesian(latitude, longitude))
                else:
                    classes.append([latitude, longitude])
            if opt.traintype == 'Classification':
                # Some imgs don't have classes, its not too many to make a difference
                try:
                    classes.append(data[row[1]['IMG_ID']])
                except:
                    fnames = fnames[:-1]
                    scenes = scenes[:-1]

    return fnames, classes, scenes

def get_im2gps3k_test(classfile="im2gps3k_places365.csv", opt=None, cartesian_coords=False):

    class_info = pd.read_csv(opt.resources + classfile)
    base_folder = opt.im2gps3k

    fnames = []
    classes = []
    scenes = []

    for row in class_info.iterrows():
        filename = base_folder + row[1]['IMG_ID']
        if exists(filename):
            fnames.append(filename)
            
            latitude = float(row[1]['LAT'])
            longitude = float(row[1]['LON'])
            # print(row[1]['LAT'])

            scenes.append([row[1]['S3_Label'], row[1]['S16_Label'], row[1]['S365_Label']])

            if cartesian_coords:
                classes.append(toCartesian(latitude, longitude))
            else:
                classes.append([latitude, longitude])
                
    
    #print(classes)
    return fnames, classes, scenes

def get_im2gps3k_test_classes(classfile="im2gps3k_places365.csv", opt=None, cartesian_coords=False):
    class_info = pd.read_csv(opt.resources + classfile)
    base_folder = opt.im2gps3k

    classes = []

    for row in class_info.iterrows():
        filename = base_folder + row[1]['IMG_ID']
        if exists(filename):          
            latitude = float(row[1]['LAT'])
            longitude = float(row[1]['LON'])

            if cartesian_coords:
                classes.append(toCartesian(latitude, longitude))
            else:
                classes.append([latitude, longitude])    
    
    return classes

def get_yfcc26k_test(classfile="yfcc25600_places365.csv", opt=None, cartesian_coords=False):
    class_info = pd.read_csv(opt.resources + classfile)
    base_folder = opt.yfcc26k

    fnames = []
    classes = []
    scenes = []

    for row in class_info.iterrows():
        filename = base_folder + row[1]['IMG_ID']
        if exists(filename):
            fnames.append(filename)
            
            latitude = float(row[1]['LAT'])
            longitude = float(row[1]['LON'])
            #print(row[1]['LAT'])

            scenes.append([row[1]['S3_Label'], row[1]['S16_Label'], row[1]['S365_Label']])

            if cartesian_coords:
                classes.append(toCartesian(latitude, longitude))
            else:
                classes.append([latitude, longitude])
    
    print("test")
    return fnames, classes, scenes

def get_yfcc26k_test_classes(classfile="yfcc25600_places365.csv", opt=None, cartesian_coords=False):
    class_info = pd.read_csv(opt.resources + classfile)
    base_folder = opt.yfcc26k

    classes = []

    for row in class_info.iterrows():
        filename = base_folder + row[1]['IMG_ID']
        if exists(filename):            
            latitude = float(row[1]['LAT'])
            longitude = float(row[1]['LON'])

            if cartesian_coords:
                classes.append(toCartesian(latitude, longitude))
            else:
                classes.append([latitude, longitude])
    
    return classes


def read_frames(fname, one_frame=False):
    path = glob.glob(fname + '/*.jpg')
    
    vid = []
    classes = []
    for img in path:
        buffer = im.open(img).convert('RGB')
        classes.append(list(float(c) for c in (img.split("/")[-1][3:-4].split("_"))))
        vid.append(buffer)
        if one_frame:
            break
    return vid, classes

class M16Dataset(Dataset):

    def __init__(self, crop_size = 112, split='train', opt=None):

        np.random.seed(0)
        
        self.split = split 
        if split == 'train':
            fnames, classes, scenes = get_mp16_train(classfile="mp16_places365.csv", opt=opt)
        if split == 'train1M':
            fnames, classes, scenes = get_mp16_train(classfile="mp16_places365_1M.csv", opt=opt)
        if split == 'train500K':
            fnames, classes, scenes = get_mp16_train(classfile="mp16_places365_500K.csv", opt=opt)
        if split == 'train100K':
            fnames, classes, scenes = get_mp16_train(classfile="mp16_places365_100K.csv", opt=opt)
        if split == 'im2gps3k':
            fnames, classes, scenes = get_im2gps3k_test(opt=opt)    
        if split == 'train3K':
            fnames, classes = get_mp16_train(classfile="mp16_places365_3K.csv", opt=opt)
        if split == 'yfcc26k':
            fnames, classes, scenes = get_yfcc26k_test(classfile="yfcc25600_places365.csv", opt=opt)
        
        if opt.hier_eval:
            maps = pickle.load(open("/home/c3-0/al209167/datasets/Resources/class_map.p", "rb"))
            self.coarse2medium = maps[0]
            self.medium2fine = maps[1]

            self.medium2fine[929] = 0
            self.medium2fine[3050] = 0
        
        # print(fnames[0])

        temp = list(zip(fnames, classes, scenes))
        np.random.shuffle(temp)
        self.fnames, self.classes, self.scenes = zip(*temp)
        self.fnames, self.classes, self.scenes = list(self.fnames), list(self.classes), list(self.scenes)

        self.data = self.fnames

        print("Loaded data, total imgs", len(fnames))
        if self.split in ['train', 'trainbdd', 'train1M', 'train500K', 'train100K']:
            self.transform = m16_transform()
        else:
            self.transform = m16_val_transform()

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = im.open(sample).convert('RGB')
        
        img = self.transform(img)


        #print(self.classes[idx])
        if self.split in ['train', 'train1M', 'trainbdd'] :
            return img, torch.Tensor(self.classes[idx]).to(torch.float64), torch.Tensor(self.scenes[idx]).to(torch.int64)
        else:
            return img, torch.Tensor(self.classes[idx]).to(torch.float64), torch.Tensor(self.scenes[idx]).to(torch.int64)

    def __len__(self):
        return len(self.data)

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    opt = getopt()

    dataset = M16Dataset(split='train100K', opt=opt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, num_workers=10, shuffle=False, drop_last=False)

    bar = tqdm(enumerate(dataloader), total = len(dataloader))
    for i, (img, classes, scenes) in bar:
        print(img.shape)
        print(classes.shape)
        print(scenes.shape)
        break
