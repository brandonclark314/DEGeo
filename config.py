import argparse
import multiprocessing
from pickle import FALSE
#import argparge
import torch

def getopt():
    parser = argparse.ArgumentParser()

    opt = parser.parse_args()
    opt.kernels = 10 #multiprocessing.cpu_count()

    opt.mp16folder = "/squash/MP-16-zstd/resources/images/mp16/"
    opt.im2gps3k = "/home/al209167/datasets/im2gps3ktest/"
    opt.yfcc26k = "/squash/MP-16-zstd/resources/images/yfcc25600/"

    opt.resources = "/home/br087771/DEGeo/"

    opt.saved_model = "/home/vi844593/DEGeo/weights/GeoCLIP100K SGD (256) LatLon RegLast 1e-6.pth"
    # opt.saved_model = "/home/vi844593/DEGeo/weights/GeoCLIP100K Adam (32) x 100 768Dim.pth"

    opt.size = 224
    opt.n_epochs = 64

    opt.description = 'GeoCLIP100K SGD (256) LatLon RegLast 1e-6'
    opt.archname = 'CLIP'
    opt.evaluate = False
    opt.scene = False
    opt.hier_eval = False

    # opt.lr = 3e-5
    # opt.lr = 5e-4
    opt.lr = 0.01
    opt.step_size = 3
    opt.partition = 'fine'

    opt.regularization_samples = 256
    opt.batch_size = 256
    opt.distances = [2500, 750, 200, 25, 1]
    opt.trainset = 'train100K'
    opt.testset = 'im2gps3k'
    opt.traintype = 'CLIP'
    opt.device = torch.device('cuda')

    return opt
