import argparse
import multiprocessing
#import argparge
import torch
import models

def getopt():
    parser = argparse.ArgumentParser()

    opt = parser.parse_args()
    opt.kernels = 16 #multiprocessing.cpu_count()

    opt.mp16folder = "/squash/MP-16-zstd/resources/images/mp16/"
    opt.im2gps3k = "/home/al209167/datasets/im2gps3ktest/"
    opt.yfcc26k = "/squash/MP-16-zstd/resources/images/yfcc25600/"

    opt.resources = "/home/br087771/DEGeo/"

    opt.saved_model = "/home/br087771/DEGeo/weights/GeoCLIP100K (No RFF)/GeoCLIP100K (No RFF)_18_0.28.pth"

    opt.size = 224
    opt.n_epochs = 32

    opt.description = 'ViT100K Hier'
    opt.archname = 'ViT'
    opt.evaluate = False
    opt.scene = False
    opt.hier_eval = True

    opt.lr = 1e-2
    opt.step_size = 3

    opt.batch_size = 32
    opt.distances = [2500, 750, 200, 25, 1]
    opt.trainset = 'train100K'
    opt.testset = 'im2gps3k'
    opt.traintype = 'Classification'
    opt.device = torch.device('cuda')

    return opt
