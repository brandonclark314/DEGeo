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


# from einops import rearrange
import csv

import json
from collections import Counter

import matplotlib.pyplot as plt
from os.path import exists

from config import getopt

def toCartesian(latitude, longitude):
    lat = latitude * np.pi / 180
    lon = longitude * np.pi / 180
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return [x, y, z]

# Need to change this to torchvision transforms 
def my_transform():
	video_transform_list = [
        RandomCrop(size=600),
        Resize(size=224),
		ClipToTensor(channel_nb=3),
		Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	]
	video_transform = Compose(video_transform_list)
	return  video_transform

def m16_transform():

    m16_transform_list = transforms.Compose([
        #transforms.RandomAffine((1, 15)),
        #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        #transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        #transforms.TenCrop(224),
        #transforms.Lambda(lambda crops: torch.stack([transforms.PILToTensor()(crop) for crop in crops])),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
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

def get_mp16_train(classfile=None, opt=None, cartesian_coords=True):

    class_info = open(opt.resources + classfile).read().splitlines()[1:]

    #print("The classes should have been", class_info['34/8d/9055806529.jpg'])
    base_folder = opt.mp16folder

    fnames = []
    classes = []

    for row in class_info:
        filename = base_folder + row.split(',')[0]
        if exists(filename):
            fnames.append(filename)
            
            latitude = float(row.split(',')[2])
            longitude = float(row.split(',')[3])
                             
            if cartesian_coords:
                classes.append(toCartesian(latitude, longitude))
            else:
                classes.append([latitude, longitude])
    

    return fnames, classes

def get_im2gps3k_test(classfile="im2gps3k_places365.csv", opt=None, cartesian_coords=False):

    class_info = pd.read_csv(opt.resources + classfile)
    base_folder = opt.im2gps3k

    fnames = []
    classes = []

    for row in class_info.iterrows():
        filename = base_folder + row[1]['IMG_ID']
        if exists(filename):
            fnames.append(filename)
            
            latitude = float(row[1]['LAT'])
            longitude = float(row[1]['LON'])
            #print(row[1]['LAT'])
    
            if cartesian_coords:
                classes.append(toCartesian(latitude, longitude))
            else:
                classes.append([latitude, longitude])
                
    
    #print(classes)
    return fnames, classes

def read_frames(fname, one_frame=False):
    path = glob.glob(fname + '/*.jpg')
    
    vid = []
    coords = []
    for img in path:
        buffer = im.open(img).convert('RGB')
        coords.append(list(float(c) for c in (img.split("/")[-1][3:-4].split("_"))))
        vid.append(buffer)
        if one_frame:
            break
    return vid, coords

class M16Dataset(Dataset):

    def __init__(self, crop_size = 112, split='train', opt=None):

        np.random.seed(0)
        
        self.split = split 
        if split == 'train':
            fnames, classes = get_mp16_train(opt=opt)
        if split == 'train1M':
            fnames, classes = get_mp16_train(classfile="mp16_places365_1M.csv", opt=opt)
        if split == 'train500K':
            fnames, classes = get_mp16_train(classfile="mp16_places365_500K.csv", opt=opt)
        if split == 'train100K':
            fnames, classes = get_mp16_train(classfile="mp16_places365_100K.csv", opt=opt)
        if split == 'im2gps3k':
            fnames, classes = get_im2gps3k_test(opt=opt)    
        

        temp = list(zip(fnames, classes))
        np.random.shuffle(temp)
        self.fnames, self.classes = zip(*temp)
        self.fnames, self.classes = list(self.fnames), list(self.classes)

        self.data = self.fnames

        print("Loaded data, total vids", len(fnames))
        if self.split in ['train', 'trainbdd']:
            self.transform = m16_transform()
        else:
            self.transform = m16_val_transform()

    def __getitem__(self, idx):

        #print(self.data[0])
        sample = self.data[idx]
        '''
        coords = []
        if not self.one_frame:
            vid, coords = read_frames(sample)
            vid = vid[:15]
            coords = coords[:15]
        else:
            vid, coords = read_frames(sample, self.one_frame)
        '''
        vid = im.open(sample).convert('RGB')
        vid = self.transform(vid)

        #print(self.classes[idx])
        if self.split in ['train', 'train1M', 'trainbdd'] :
            return vid, torch.Tensor(self.classes[idx]).to(torch.float64)
        else:
            return vid, torch.Tensor(self.classes[idx])

    def __len__(self):
        return len(self.data)

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    opt = getopt()

    dataset = M16Dataset(split='train100K', opt=opt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, num_workers=10, shuffle=False, drop_last=False)

    for i, (img, classes) in enumerate(dataloader):
        print(img.shape)
        print(classes.shape)
        break
