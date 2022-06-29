import argparse
import multiprocessing
#import argparge
import torch
import models

def getopt():
    parser = argparse.ArgumentParser()

    opt = parser.parse_args()
    opt.kernels = multiprocessing.cpu_count()

    opt.mp16folder = "/squash/MP-16-zstd/resources/images/mp16/"
    opt.im2gps3k = "/home/al209167/datasets/im2gps3ktest/"

    opt.resources = "/home/br087771/DEGeo/"

    opt.size = 224
    opt.n_epochs = 32

    opt.description = 'GeoCLIP-100K (2D GPS)'
    opt.archname = 'GeoCLIP'
    opt.evaluate = False

    opt.lr = 5e-6
    opt.step_size = 3

    opt.batch_size = 256
    opt.distances = [2500, 750, 200, 25, 1]
    opt.trainset = 'train100K'
    opt.testset = 'im2gps3k'
    opt.device = torch.device('cuda')

    return opt 
